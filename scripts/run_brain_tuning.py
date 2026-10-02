"""Version B: brain-tune a LoRA LLM with word-level EEG targets; text-only inference.

Arms (same model, initial weights, batch order, folds):
  text_only       auxiliary weight 0
  eeg             auxiliary head predicts each word's reader-averaged EEG components
  shuffled_eeg    EEG targets shuffled across training words
  random_targets  Gaussian targets of the same size
EEG helps only if `eeg` beats `shuffled_eeg` and `random_targets`.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.brainshaping.brain_tuning import BrainTunedClassifier, aux_r2, run_brain_fold  # noqa: E402
from src.brainshaping.data import arm_targets, fit_targets, item_eeg, sentence_table  # noqa: E402
from src.brainshaping.encoding import center_by_group  # noqa: E402
from src.fusion.train import TrainConfig, precision_for  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402
from src.neurolm.config import load_config, run_manifest, save_json  # noqa: E402
from src.neurolm.evaluation import (  # noqa: E402
    SENTIMENT_CLASSES, cluster_bootstrap, compute_metrics, paired_comparison,
)
from src.neurolm.splits import make_splits  # noqa: E402

COMPARISONS = [("eeg", "shuffled_eeg"), ("eeg", "random_targets"), ("eeg", "text_only")]


def load_language_model(name, revision, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype)
    return lm.to(device=device, dtype=dtype), tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/brain_tuning.yaml")
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-tag", default="brain_tuning_v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--folds", type=int, default=1, help="number of folds (start with 1; up to 5)")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--base-dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true", help="small model, 1 epoch, text_only vs eeg")
    parser.add_argument("--targets", default="centered", choices=["centered", "raw"],
                        help="centered: each sentence's mean removed from its words' EEG (the part text predicts)")
    return parser.parse_args()


def resolve_dtype(name, device):
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def per_sentence(values, items, sentences, k):
    """Scatter item rows back to per-sentence [n_words, k] arrays (NaN for words without EEG)."""
    row_of = {sid: i for i, sid in enumerate(sentences["sentence_id"])}
    out = [np.full((len(w), k), np.nan, dtype=np.float32) for w in sentences["words"]]
    for value, sid, word in zip(values, items["sentence_id"], items["word_index"]):
        out[row_of[sid]][word] = value
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or (cfg["quick_model"] if args.quick else cfg["base_model"])
    revision = cfg["base_model_revision"] if model_name == cfg["base_model"] else None
    arms = args.arms or (["text_only", "eeg"] if args.quick else cfg["arms"])
    train_cfg = TrainConfig(**cfg["train"])
    if args.epochs is not None or args.quick:
        train_cfg.epochs = args.epochs if args.epochs is not None else 1.0
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_dtype(args.base_dtype or cfg.get("base_dtype", "auto"), device)
    aux = cfg["aux"]

    trials = load_word_eeg(args.word_eeg_dir)
    sentences = sentence_table(trials)
    items, eeg = item_eeg(trials)
    if args.targets == "centered":
        codes = np.unique(items["sentence_id"].to_numpy(), return_inverse=True)[1]
        eeg = center_by_group(eeg.astype(np.float64), codes).astype(np.float32)
    words = sentences["words"].tolist()
    labels = sentences["label_id"].to_numpy()
    samples = pd.DataFrame({"sample_id": sentences["sentence_id"].astype(str), "subject_id": "all",
                            "sentence_id": sentences["sentence_id"], "label_id": labels})
    split_cfg = cfg["split"]
    splits = make_splits(split_cfg["protocol"], samples, split_cfg["seed"], split_cfg["params"])[:args.folds]
    print(f"{len(sentences)} sentences, {len(items)} word items with EEG; {len(splits)} fold(s); arms {arms}; "
          f"{model_name} ({dtype}) on {device}")

    lm, tokenizer = load_language_model(model_name, revision, dtype, device)
    n_layers = lm.config.num_hidden_layers
    layer = min(aux["layer"], n_layers)
    model = BrainTunedClassifier(lm, tokenizer, class_words=cfg["class_words"], prompt=cfg["prompt"],
                                 lora=cfg["lora"], aux_dim=aux["k"], aux_layer=layer).to(device)
    init = model.trainable_state()
    run_dir = os.path.join(args.results_dir, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    settings = {"model": model_name, "revision": revision, "dtype": str(dtype), "train": train_cfg.to_dict(),
                "aux": {**aux, "layer": layer}, "prompt": cfg["prompt"], "lora": cfg["lora"], "split": split_cfg,
                "n_sentences": int(len(sentences)), "targets": args.targets}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]
    save_json(run_manifest(cfg, {"settings": settings, "settings_key": key, "arms": arms}),
              os.path.join(run_dir, "manifest.json"))

    collected = {arm: [] for arm in arms}
    aux_scores = {arm: [] for arm in arms}
    dtype_eval, _ = precision_for(device)
    started = time.time()
    for k, split in enumerate(splits):
        train_ids = set(samples["sentence_id"].iloc[split.train])
        item_train = np.flatnonzero(items["sentence_id"].isin(train_ids).to_numpy())
        transform, _ = fit_targets(eeg[item_train], aux["k"])
        real = transform(eeg)
        real_by_sentence = per_sentence(real, items, sentences, aux["k"])
        scores = {}
        for arm in arms:
            csv_path = os.path.join(run_dir, arm, f"fold_{k}.csv")
            meta_path = os.path.join(run_dir, arm, f"fold_{k}.json")
            if os.path.exists(csv_path) and os.path.exists(meta_path) and json.load(open(meta_path))["key"] == key:
                frame = pd.read_csv(csv_path)
                aux_scores[arm].append(json.load(open(meta_path)).get("aux_r2_real_test"))
                print(f"{arm} fold {k + 1}: reuse saved predictions")
            else:
                rng = np.random.default_rng(split_cfg["seed"] * 1000 + k)
                weight = 0.0 if arm == "text_only" else aux["weight"]
                arm_values = real if arm == "text_only" else arm_targets(arm, real, item_train, rng)
                targets = per_sentence(arm_values, items, sentences, aux["k"])
                print(f"{arm} fold {k + 1}/{len(splits)}: train {len(split.train)}, val {len(split.val)}, "
                      f"test {len(split.test)} sentences")
                probs, info = run_brain_fold(model, init, words, labels, targets, split, weight, train_cfg, device)
                os.makedirs(os.path.join(run_dir, arm), exist_ok=True)
                torch.save(model.trainable_state(), os.path.join(run_dir, arm, f"fold_{k}_weights.pt"))
                r2 = None if arm == "text_only" else aux_r2(model, split.test, words, real_by_sentence,
                                                            device, dtype_eval)
                predicted = probs.argmax(1)
                ids = samples["sentence_id"].iloc[split.test].to_numpy()
                frame = pd.DataFrame({"sample_id": ids.astype(str), "sentence_id": ids, "seed": 0,
                                      "true_id": labels[split.test], "predicted_id": predicted, "fold": k, "arm": arm,
                                      "true_label": [SENTIMENT_CLASSES[i] for i in labels[split.test]],
                                      "predicted_label": [SENTIMENT_CLASSES[i] for i in predicted]})
                for i, name in enumerate(SENTIMENT_CLASSES):
                    frame[f"prob_{name}"] = probs[:, i]
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                frame.to_csv(csv_path, index=False)
                save_json({"key": key, "info": info, "aux_r2_real_test": r2}, meta_path)
                aux_scores[arm].append(r2)
            scores[arm] = compute_metrics(frame["true_id"], frame["predicted_id"])["macro_f1"]
            collected[arm].append(frame)
        print(f"fold {k + 1} test macro-F1: " + ", ".join(f"{a} {v:.3f}" for a, v in scores.items()))

    predictions = {arm: pd.concat(frames, ignore_index=True) for arm, frames in collected.items()}
    summary = {"model": model_name, "dtype": str(dtype), "aux": settings["aux"], "n_folds": len(splits),
               "n_test_sentences": int(len(next(iter(predictions.values())))), "arms": {}, "comparisons": [],
               "runtime_s": time.time() - started}
    for arm, frame in predictions.items():
        r2 = [v for v in aux_scores[arm] if v is not None]
        summary["arms"][arm] = {"metrics": compute_metrics(frame["true_id"], frame["predicted_id"]),
                                "bootstrap": cluster_bootstrap(frame, "sentence_id", n_boot=2000),
                                "aux_r2_real_test_mean": float(np.mean(r2)) if r2 else None}
    for a, b in COMPARISONS:
        if a in predictions and b in predictions:
            summary["comparisons"].append({"candidate": a, "baseline": b,
                                           **paired_comparison(predictions[b], predictions[a], cluster="sentence_id",
                                                               n_boot=2000, n_perm=5000)})
    save_json(summary, os.path.join(run_dir, "summary.json"))
    lines = ["# Brain-tuned LoRA LLM (EEG as training signal) — results", "",
             f"{model_name}, auxiliary head on layer {layer}, {aux['k']} EEG components, weight {aux['weight']}; "
             f"{len(splits)} fold(s), {summary['n_test_sentences']} unseen test sentences. Text-only inference.", "",
             "| arm | macro-F1 | 95% CI | accuracy | aux R² on real held-out EEG |", "|---|---:|---|---:|---:|"]
    for arm, values in summary["arms"].items():
        ci = values["bootstrap"]["macro_f1"]["ci95"]
        r2 = values["aux_r2_real_test_mean"]
        lines.append(f"| {arm} | {values['metrics']['macro_f1']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{values['metrics']['accuracy']:.3f} | {'—' if r2 is None else f'{r2:.3f}'} |")
    lines += ["", "| comparison | Δ macro-F1 | 95% CI | p |", "|---|---:|---|---:|"]
    for c in summary["comparisons"]:
        lines.append(f"| {c['candidate']} − {c['baseline']} | {c['delta_macro_f1']:.3f} | "
                     f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {c['p_value_two_sided']:.3f} |")
    lines += ["", "EEG helps only if `eeg` beats `shuffled_eeg` and `random_targets` (CI above 0). "
              "The aux R² column checks that the model actually learned to predict real EEG."]
    with open(os.path.join(run_dir, "brain_tuning_results.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
