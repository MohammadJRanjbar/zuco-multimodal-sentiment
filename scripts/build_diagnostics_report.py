"""Combine stage 1-4 outputs into reports/eeg_text_diagnostics.md.

Each stage is optional: missing folders are reported as "not run". The
"where it fails" section applies fixed rules (p < 0.05 against the
permutation null) to the saved numbers; it is a guide for the next experiment,
not a substitute for reading the tables.
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

ALPHA = 0.05


def load(path, reader=pd.read_csv):
    if not path or not os.path.exists(path):
        return None
    try:
        return reader(path)
    except pd.errors.EmptyDataError:
        return None


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def table(frame, columns):
    if frame is None or frame.empty:
        return "*not available*"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def significant(frame, level, target):
    if frame is None:
        return None
    part = frame[(frame["level"] == level) & (frame["target"] == target)]
    if part.empty or pd.isna(part["p_value"].iloc[0]):
        return None
    score, metric = part["score"].iloc[0], part["metric"].iloc[0]
    chance = {"R2": 0.0, "AUC": 0.5}.get(metric, part["null_mean"].iloc[0])
    return bool(part["p_value"].iloc[0] < ALPHA and score > part["null_q95"].iloc[0] and score > chance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", default=None, help="output of analyze_eeg_signal.py")
    parser.add_argument("--fusion-dir", default=None, help="output of analyze_fusion_model.py")
    parser.add_argument("--errors-dir", default=None, help="output of analyze_errors.py")
    parser.add_argument("--lora-report", default=None, help="eeg_text_lora_results.md of the run")
    parser.add_argument("--report", default="reports/eeg_text_diagnostics.md")
    args = parser.parse_args()

    lines = ["# Where does EEG + text sentiment fail? Diagnostics", ""]
    findings, fixes = [], []

    # Stage 1
    lines += ["## Stage 1 — signal in word-level EEG", ""]
    summary = load(os.path.join(args.signal_dir, "stage1_summary.json"), lambda p: json.load(open(p))) if args.signal_dir else None
    probes = load(os.path.join(args.signal_dir, "word_probes.csv")) if args.signal_dir else None
    if summary:
        variance = summary["variance_components_median"]
        rel = summary["reliability"]
        lines += [
            f"* {summary['n_word_observations']:,} word observations ({summary['n_items']:,} sentence-word items), "
            f"{summary['feature_info']['n_bands']} bands × {summary['feature_info']['n_channels']} electrodes.",
            f"* Median variance share per feature: reader {fmt(variance['reader'])}, word item {fmt(variance['word_item'])}, "
            f"residual {fmt(variance['residual'])}.",
            f"* Cross-reader reliability (split-half, projected to all readers): median {fmt(rel['median_all_readers'])}, "
            f"90th percentile {fmt(rel['q90_all_readers'])}; implied single-reader reliability {fmt(rel['median_single_reader'])}.",
            f"* By band (median, all readers): " + ", ".join(f"{b} {fmt(v)}" for b, v in rel["median_all_readers_by_band"].items()),
            "", table(probes, ["level", "target", "kind", "metric", "score", "null_q95", "p_value"]), "",
            "Plots: `plots/reliability_all_readers.png`, `plots/correlation_<target>.png` in the stage 1 folder.", "",
        ]
        if probes is not None and "n_perm" in probes and (1 / (probes["n_perm"] + 1) >= ALPHA).all():
            findings.append("Too few permutations to reach p < 0.05; rerun stage 1 with --n-perm >= 50 and "
                            "--n-perm-single >= 20 before reading the significance-based findings.")
        controls = [t for t in ["length", "zipf", "surprisal", "trt_ms", "is_content"]
                    if significant(probes, "reader-averaged", t)]
        # in_lexicon (emotional word yes/no) is confounded with word class, length, and frequency,
        # which EEG does track; it is reported in the table but never counted as sentiment evidence.
        sentiment_avg = [t for t in ["valence", "abs_valence", "sentence_label"]
                         if significant(probes, "reader-averaged", t)]
        sentiment_single = [t for t in ["valence", "abs_valence", "sentence_label"]
                            if significant(probes, "single reader", t)]
        n_tests = int(probes["p_value"].notna().sum()) if probes is not None else 0
        if n_tests:
            lines += [f"*{n_tests} probes were tested at p < {ALPHA}; about {n_tests * ALPHA:.0f} false positives are "
                      "expected by chance. `in_lexicon` is confounded with word class, length, and frequency.*", ""]
        if variance["reader"] > variance["word_item"]:
            findings.append("Reader identity explains more EEG variance than the words being read.")
            fixes.append("Remove reader variance before fusion: per-reader alignment, reader-adversarial loss.")
        if rel["median_all_readers"] < 0.2:
            findings.append(f"Word-level EEG is weakly reliable even averaged over readers (median {fmt(rel['median_all_readers'])}).")
            fixes.append("Average over readers and use only the most reliable bands/electrodes.")
        if controls and not sentiment_avg:
            findings.append(f"EEG predicts reading-related positive controls ({', '.join(controls)}) but no sentiment target: "
                            "the pipeline can detect EEG effects, sentiment information is absent or below detection.")
            fixes.append("The ceiling is in the data/labels: treat sentiment fusion gains as unlikely; consider "
                         "word-level reading targets, eye tracking, or EEG as an uncertainty signal (stage 4).")
        if not controls:
            findings.append("Even positive controls are not decodable: check the extraction and preprocessing first.")
        if sentiment_avg and not sentiment_single:
            findings.append(f"Sentiment targets ({', '.join(sentiment_avg)}) are decodable only after averaging over readers.")
            fixes.append("Feed reader-averaged word EEG tokens to the fusion model.")
        if sentiment_single:
            findings.append(f"Single-reader EEG carries sentiment-related information ({', '.join(sentiment_single)}).")
    else:
        lines += ["*not run*", ""]
    covariates = load(os.path.join(args.signal_dir, "valence_beyond_covariates.csv")) if args.signal_dir else None
    if covariates is not None:
        lines += ["Valence of lexicon words — does EEG add beyond word length, frequency, surprisal, position, word class?", "",
                  table(covariates, ["model", "target", "R2"]), ""]
    stage2 = load(os.path.join(args.signal_dir, "stage2_representations.csv")) if args.signal_dir else None
    lines += ["## Stage 2 — which sentence-level representation keeps the information?", "",
              table(stage2, ["representation", "target", "metric", "score", "null_q95", "p_value"]), ""]
    if stage2 is not None:
        sentiment = stage2[(stage2["target"] == "sentence_label") & (stage2["p_value"] < ALPHA)]
        if sentiment.empty:
            findings.append("No sentence-level EEG representation decodes the sentiment label above its permutation null.")
        else:
            findings.append("Sentence sentiment is decodable from: " + ", ".join(sentiment["representation"]))

    # Stage 3
    lines += ["## Stage 3 — does the fusion model read its EEG tokens?", ""]
    stage3 = load(os.path.join(args.fusion_dir, "stage3_summary.json"), lambda p: json.load(open(p))) if args.fusion_dir else None
    if stage3:
        reliance = pd.DataFrame(stage3["reliance"])
        share = stage3["attribution_mean_share"]
        lines += [f"Arm `{stage3['arm']}`, fold {stage3['fold']}, {stage3['n_trials']} held-out trials.", "",
                  table(reliance, ["variant", "accuracy", "flip_rate_vs_aligned", "mean_abs_logit_change"]), "",
                  f"Gradient × input share of the decision: EEG tokens {fmt(share.get('eeg'))}, word tokens {fmt(share.get('word'))}, "
                  f"prompt {fmt(share.get('prompt'))}; EEG per-token / word per-token = "
                  f"{fmt(share.get('eeg_per_token_over_word_per_token'))}.", "",
                  "Attention by layer: `attention_by_layer.png` in the stage 3 folder.", ""]
        slot = pd.DataFrame(stage3.get("slot_probes", []))
        if not slot.empty:
            lines += [table(slot, ["space", "target", "metric", "score", "null_q95", "p_value"]), "",
                      "*Only `raw EEG input` and `projector output` isolate EEG. Hidden states at an EEG slot also "
                      "attend to the word just before it, so hidden-layer probes mix text and EEG information.*", ""]
        shuffled = reliance.set_index("variant").get("accuracy", pd.Series()).get("shuffled_within_reader")
        aligned = reliance.set_index("variant").get("accuracy", pd.Series()).get("aligned")
        flips = reliance.set_index("variant").get("flip_rate_vs_aligned", pd.Series()).get("shuffled_within_reader")
        if flips is not None and flips < 0.02:
            findings.append(f"The trained model barely reacts to EEG: shuffling it changes {100 * flips:.1f}% of predictions.")
            fixes.append("Force EEG use: word dropout (hide words so EEG must carry them), contrastive EEG–word "
                         "pretraining of the projector, or an auxiliary EEG-prediction loss.")
        elif aligned is not None and shuffled is not None and aligned <= shuffled:
            findings.append("The model reacts to EEG, but aligned EEG is not more accurate than shuffled EEG: "
                            "it uses EEG in a non-specific way.")
        if not slot.empty:
            raw = slot[(slot["space"] == "raw EEG input") & (slot["target"] == "valence")]
            hidden = slot[slot["space"].str.startswith("hidden") & (slot["target"] == "valence")]
            if not raw.empty and not hidden.empty and raw["p_value"].iloc[0] < ALPHA and (hidden["p_value"] >= ALPHA).all():
                findings.append("Valence information present in the raw EEG input is lost inside the model.")
                fixes.append("Align the projector to word semantics before sentiment training (contrastive pretraining).")
    else:
        lines += ["*not run*", ""]

    # Stage 4
    lines += ["## Stage 4 — where the text model fails, and EEG there", ""]
    stage4 = load(os.path.join(args.errors_dir, "stage4_summary.json"), lambda p: json.load(open(p))) if args.errors_dir else None
    subsets = load(os.path.join(args.errors_dir, "subset_effects.csv")) if args.errors_dir else None
    if stage4:
        lines += [f"Text-only sentence accuracy {fmt(stage4['text_only_sentence_accuracy'])}; "
                  f"{stage4['n_text_errors']} sentences misclassified.", "",
                  table(subsets, ["subset", "comparison", "n_sentences", "accuracy_a", "accuracy_b", "delta", "ci95"]), ""]
        error = stage4.get("eeg_predicts_text_errors", {}).get("text_wrong_from_eeg")
        if error and not error.get("skipped") and error["p_value"] >= ALPHA:
            findings.append(f"EEG does not predict which sentences the text model gets wrong (AUC {fmt(error['score'])}).")
        if error and not error.get("skipped"):
            lines.append(f"EEG predicts which sentences the text model gets wrong: AUC {fmt(error['score'])} "
                         f"(null 95th percentile {fmt(error['null_q95'])}, p = {fmt(error['p_value'])}).")
            if error["p_value"] < ALPHA:
                findings.append("EEG predicts where the text model fails.")
                fixes.append("Use EEG as a confidence / routing signal rather than as extra features.")
        # Subsets selected on the text model's own errors or confidence favour any model that differs from it
        # (selection / regression to the mean), so they never count as evidence that EEG helps.
        lines += ["*`text_wrong` and `text_low_confidence` are selected on the text model's own errors: any other "
                  "model looks better there. Treat effects in these rows as hypotheses for other folds.*", ""]
        helpful = subsets[(subsets["comparison"] == "text_eeg - text_shuffled_eeg")
                          & (subsets["n_sentences"] >= 10)
                          & ~subsets["subset"].isin(["text_wrong", "text_low_confidence"])
                          & subsets["ci95"].astype(str).str.match(r"\[[0-9]")] if subsets is not None else None
        if helpful is not None and not helpful.empty:
            findings.append("Aligned EEG beats shuffled EEG within: " + ", ".join(helpful["subset"]))
    else:
        lines += ["*not run*", ""]

    if args.lora_report and os.path.exists(args.lora_report):
        lines += ["## Fusion run summary", "", open(args.lora_report).read(), ""]

    lines += ["## Where it fails and what to try", ""]
    lines += [f"* {f}" for f in findings] or ["* No rule fired; read the tables."]
    lines += ["", "Suggested next steps:", ""]
    lines += [f"* {f}" for f in dict.fromkeys(fixes)] or ["* —"]
    lines += ["", "*Rules use p < 0.05 against label-permutation nulls; they guide the next experiment and should be "
                  "read with the tables.*"]
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    for folder in (args.signal_dir, args.fusion_dir, args.errors_dir):
        if folder and os.path.isdir(folder):
            shutil.copy(args.report, os.path.join(folder, "eeg_text_diagnostics.md"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
