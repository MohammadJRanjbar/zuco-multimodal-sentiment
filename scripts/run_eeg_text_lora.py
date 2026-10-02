"""Word-aligned EEG + text sentiment with a LoRA-tuned LLM, against matched controls.

Arms (same model, initial weights, batch order, and folds; only the EEG input differs):
  text_only           sentence only, no EEG tokens
  text_eeg            each word followed by the reader's EEG for that word
  text_shuffled_eeg   same, but EEG vectors re-assigned among the reader's words
                      in the same partition (alignment destroyed)
  text_fixation_only  EEG tokens carry only whether the word was fixated

Predeclared verdict: the model uses EEG only if text_eeg beats
text_shuffled_eeg (paired sentence-cluster CI above 0 and p < alpha).
Resumable: finished arm/fold predictions are reused when the settings match.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress in Colab before any crash

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.fusion.data import ARM_CONTROL, build_fusion_data, model_inputs, reader_statistics  # noqa: E402
from src.fusion.model import FusionClassifier  # noqa: E402
from src.fusion.train import TrainConfig, run_fold  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402
from src.neurolm.config import load_config, run_manifest, save_json  # noqa: E402
from src.neurolm.evaluation import (  # noqa: E402
    SENTIMENT_CLASSES, cluster_bootstrap, compute_metrics, paired_comparison, sentence_aggregate,
)
from src.neurolm.splits import make_splits  # noqa: E402

COMPARISONS = [
    ("text_eeg", "text_shuffled_eeg", "does aligned EEG add anything? (primary)"),
    ("text_eeg", "text_only", "EEG model vs text alone"),
    ("text_eeg", "text_fixation_only", "EEG content beyond fixation pattern"),
    ("text_fixation_only", "text_only", "fixation pattern vs text alone"),
]


def load_language_model(name, revision, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype)
    return lm.to(device=device, dtype=dtype), tokenizer


def resolve_dtype(name, device):
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/eeg_text_lora.yaml")
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-tag", default="eeg_text_lora_v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=None, help="run only the first N folds")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--base-dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", default="reports/eeg_text_lora_results.md")
    parser.add_argument("--quick", action="store_true", help="small model, 1 fold, 1 epoch, text_only vs text_eeg")
    parser.add_argument("--no-save-weights", action="store_true",
                        help="do not save trained LoRA + projector weights (needed for the fusion diagnostics)")
    parser.add_argument("--retrain-missing-weights", action="store_true",
                        help="retrain folds whose predictions exist but whose weights were not saved")
    return parser.parse_args()


def predictions_table(data, rows, probs, arm, fold, seed):
    samples = data.samples.iloc[rows]
    predicted = probs.argmax(axis=1)
    frame = pd.DataFrame({
        "sample_id": samples["sample_id"].to_numpy(),
        "subject_id": samples["subject_id"].to_numpy(),
        "sentence_id": samples["sentence_id"].to_numpy(),
        "true_label": [SENTIMENT_CLASSES[i] for i in samples["label_id"]],
        "predicted_label": [SENTIMENT_CLASSES[i] for i in predicted],
    })
    for i, name in enumerate(SENTIMENT_CLASSES):
        frame[f"prob_{name}"] = probs[:, i]
    frame["true_id"] = samples["label_id"].to_numpy()
    frame["predicted_id"] = predicted
    frame["fold"] = fold
    frame["arm"] = arm
    frame["seed"] = seed
    return frame


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def write_report(path, run_dir, summary):
    s = summary
    lines = [
        "# Word-aligned EEG + text sentiment (LoRA LLM) — results", "",
        f"* Model: `{s['model']}` + LoRA ({s['lora']['r']}/{s['lora']['alpha']} on {', '.join(s['lora']['targets'])}),"
        f" {s['trainable_parameters']:,} trainable parameters, {s['dtype']}",
        f"* Data: {s['n_trials']} reader × sentence trials, {s['n_readers']} readers, {s['n_sentences']} sentences;"
        f" {100 * s['fraction_words_with_eeg']:.1f}% of words have fixation-locked EEG ({s['eeg_measure']} band power,"
        f" log={s['log_transformed']}, per-reader standardization on training sentences)",
        f"* Split: {s['protocol']} protocol (unseen sentences), {s['n_folds']} fold(s), seed {s['seed']}",
        "", "## Arms", "",
        "| arm | macro-F1 (trial) | 95% CI (sentences) | accuracy | macro-F1 averaged over readers |",
        "|---|---:|---|---:|---:|",
    ]
    for arm, m in s["arms"].items():
        ci = m["bootstrap"]["macro_f1"]["ci95"]
        lines.append(f"| {arm} | {fmt(m['trial']['macro_f1'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"{fmt(m['trial']['accuracy'])} | {fmt(m['sentence_level']['macro_f1']['mean'])} |")
    lines += ["", "## Paired comparisons (same trials; sentence-cluster bootstrap and permutation)", "",
              "| comparison | question | Δ macro-F1 | 95% CI | p | same predictions |", "|---|---|---:|---|---:|---:|"]
    for c in s["comparisons"]:
        lines.append(f"| {c['candidate']} − {c['baseline']} | {c['question']} | {fmt(c['delta_macro_f1'])} | "
                     f"[{fmt(c['ci95'][0])}, {fmt(c['ci95'][1])}] | {fmt(c['p_value_two_sided'])} | "
                     f"{100 * c['agreement']:.1f}% |")
    lines += ["", "## Verdict", "", s["verdict"], ""]
    text = "\n".join(lines) + "\n"
    for target in (path, os.path.join(run_dir, "eeg_text_lora_results.md")):
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w") as handle:
            handle.write(text)
    return text


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or (cfg["quick_model"] if args.quick else cfg["base_model"])
    revision = cfg["base_model_revision"] if model_name == cfg["base_model"] else None
    arms = args.arms or (["text_only", "text_eeg"] if args.quick else cfg["arms"])
    train_cfg = TrainConfig(**cfg["train"])
    if args.epochs is not None or args.quick:
        train_cfg.epochs = args.epochs if args.epochs is not None else 1.0
    n_folds = args.folds or (1 if args.quick else None)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_dtype(args.base_dtype or cfg.get("base_dtype", "auto"), device)

    trials = load_word_eeg(args.word_eeg_dir)
    data = build_fusion_data(trials, log_power=cfg["eeg"].get("log_power", "auto"))
    samples = data.samples
    split_cfg = cfg["split"]
    splits = make_splits(split_cfg["protocol"], samples, split_cfg["seed"], split_cfg["params"])[:n_folds]
    words_total = int(samples["n_words"].sum())
    print(f"{len(samples)} trials, {samples['subject_id'].nunique()} readers, {samples['sentence_id'].nunique()} "
          f"sentences, {samples['n_words_with_eeg'].sum() / words_total:.1%} of words with EEG; "
          f"{len(splits)} fold(s); arms {arms}; model {model_name} ({dtype}) on {device}")

    lm, tokenizer = load_language_model(model_name, revision, dtype, device)
    model = FusionClassifier(lm, tokenizer, class_words=cfg["class_words"], eeg_dim=data.dim,
                             prompt=cfg["prompt"], lora=cfg["lora"], projector=cfg["projector"]).to(device)
    init = model.trainable_state()
    run_dir = os.path.join(args.results_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    if args.quick and args.report == "reports/eeg_text_lora_results.md":
        args.report = os.path.join(run_dir, "quick_report.md")  # never overwrite the main report
    settings = {"model": model_name, "revision": revision, "dtype": str(dtype), "train": train_cfg.to_dict(),
                "prompt": cfg["prompt"], "lora": cfg["lora"], "projector": cfg["projector"], "split": split_cfg,
                "eeg": cfg["eeg"], "n_samples": int(len(samples))}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]
    save_json(run_manifest(cfg, {"settings": settings, "settings_key": key, "arms": arms,
                                 "trainable_parameters": model.n_trainable()}),
              os.path.join(run_dir, "manifest.json"))

    # Fold-major order: after each fold every arm has been trained on it, so a
    # first aligned-vs-shuffled comparison (and stage 3) is available early.
    collected = {arm: [] for arm in arms}
    started = time.time()
    for k, split in enumerate(splits):
        stats, fallback = reader_statistics(data, split.train)
        fold_scores = {}
        for arm in arms:
            control = ARM_CONTROL[arm]
            csv_path = os.path.join(run_dir, arm, f"fold_{k}.csv")
            meta_path = os.path.join(run_dir, arm, f"fold_{k}.json")
            weights_path = os.path.join(run_dir, arm, f"fold_{k}_weights.pt")
            missing_weights = args.retrain_missing_weights and not os.path.exists(weights_path)
            if (os.path.exists(csv_path) and os.path.exists(meta_path) and not missing_weights
                    and json.load(open(meta_path))["key"] == key):
                frame = pd.read_csv(csv_path, dtype={"sample_id": str})  # ids are strings in fresh runs
                print(f"{arm} fold {k + 1}: reuse saved predictions")
            else:
                rng = np.random.default_rng(split_cfg["seed"] * 1000 + k)
                inputs = model_inputs(data, stats, control or "aligned", split, rng)
                print(f"{arm} fold {k + 1}/{len(splits)}: train {len(split.train)}, val {len(split.val)}, "
                      f"test {len(split.test)}")
                probs, info = run_fold(model, init, data, inputs, split, control is not None, train_cfg, device)
                if not args.no_save_weights:
                    os.makedirs(os.path.join(run_dir, arm), exist_ok=True)
                    torch.save(model.trainable_state(), weights_path)
                frame = predictions_table(data, split.test, probs, arm, k, split_cfg["seed"])
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                frame.to_csv(csv_path, index=False)
                save_json({"key": key, "info": info, "transductive_reader_norm": fallback}, meta_path)
            fold_scores[arm] = compute_metrics(frame["true_id"], frame["predicted_id"])["macro_f1"]
            collected[arm].append(frame)
        print(f"fold {k + 1} test macro-F1: " + ", ".join(f"{a} {v:.3f}" for a, v in fold_scores.items()))
    predictions = {arm: pd.concat(frames, ignore_index=True) for arm, frames in collected.items()}

    summary = {
        "model": model_name, "dtype": str(dtype), "lora": cfg["lora"], "trainable_parameters": model.n_trainable(),
        "n_trials": int(len(samples)), "n_readers": int(samples["subject_id"].nunique()),
        "n_sentences": int(samples["sentence_id"].nunique()),
        "fraction_words_with_eeg": float(samples["n_words_with_eeg"].sum() / words_total),
        "eeg_measure": cfg["eeg"]["measure"], "log_transformed": data.log_transformed,
        "protocol": split_cfg["protocol"], "n_folds": len(splits), "seed": split_cfg["seed"],
        "arms": {}, "comparisons": [], "runtime_s": time.time() - started,
    }
    for arm, frame in predictions.items():
        summary["arms"][arm] = {
            "trial": compute_metrics(frame["true_id"], frame["predicted_id"]),
            "sentence_level": sentence_aggregate(frame),
            "bootstrap": cluster_bootstrap(frame, "sentence_id", n_boot=cfg["bootstrap"]["n_boot"]),
        }
    for candidate, baseline, question in COMPARISONS:
        if candidate in predictions and baseline in predictions:
            a, b = predictions[candidate], predictions[baseline]
            result = paired_comparison(b, a, cluster="sentence_id", n_boot=cfg["bootstrap"]["n_boot"],
                                       n_perm=cfg["bootstrap"]["n_perm"])
            merged = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
            agreement = float((merged["predicted_id_a"] == merged["predicted_id_b"]).mean())
            summary["comparisons"].append({"candidate": candidate, "baseline": baseline, "question": question,
                                           "agreement": agreement, **result})
    primary = next((c for c in summary["comparisons"]
                    if c["candidate"] == "text_eeg" and c["baseline"] == "text_shuffled_eeg"), None)
    eeg = summary["arms"].get("text_eeg", {}).get("trial", {}).get("macro_f1")
    text = summary["arms"].get("text_only", {}).get("trial", {}).get("macro_f1")
    if primary is None:
        verdict = (f"Primary comparison not run (arms: {', '.join(predictions)}). text+EEG macro-F1 {fmt(eeg)}, "
                   f"text only {fmt(text)}; without the shuffled-EEG arm a difference cannot be attributed to EEG.")
    elif primary["ci95"][0] > 0 and primary["p_value_two_sided"] < cfg["decision"]["alpha"]:
        verdict = (f"**The model uses aligned EEG.** Real word-aligned EEG beats shuffled EEG by "
                   f"{fmt(primary['delta_macro_f1'])} macro-F1 (95% CI {fmt(primary['ci95'][0])} to "
                   f"{fmt(primary['ci95'][1])}, p = {fmt(primary['p_value_two_sided'])}).")
    else:
        verdict = (f"**No reliable evidence that the model uses aligned EEG.** Real vs shuffled EEG: Δ = "
                   f"{fmt(primary['delta_macro_f1'])} macro-F1 (95% CI {fmt(primary['ci95'][0])} to "
                   f"{fmt(primary['ci95'][1])}, p = {fmt(primary['p_value_two_sided'])}). ")
        versus_text = next((c for c in summary["comparisons"]
                            if c["candidate"] == "text_eeg" and c["baseline"] == "text_only"), None)
        if versus_text and versus_text["ci95"][0] <= 0 <= versus_text["ci95"][1]:
            verdict += f"The text+EEG model ({fmt(eeg)}) is statistically indistinguishable from text alone ({fmt(text)})."
        elif versus_text:
            verdict += (f"The text+EEG model ({fmt(eeg)}) differs from text alone ({fmt(text)}), but the shuffled-EEG "
                        "control does not show that the difference is specific to aligned EEG.")
    summary["verdict"] = verdict
    save_json(summary, os.path.join(run_dir, "summary.json"))
    print(write_report(args.report, run_dir, summary))


if __name__ == "__main__":
    main()
