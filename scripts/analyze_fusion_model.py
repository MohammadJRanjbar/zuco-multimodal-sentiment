"""Stage 3: look inside a trained EEG + text LoRA model (one arm, one fold).

Needs the LoRA run folder with saved weights (scripts/run_eeg_text_lora.py
saves ``<arm>/fold_<k>_weights.pt``) and, for the representation probes, the
``word_targets.csv`` written by scripts/analyze_eeg_signal.py. Use a GPU.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress in Colab before any crash

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.diagnostics import fusion  # noqa: E402
from src.fusion.data import build_fusion_data, model_inputs, reader_statistics  # noqa: E402
from src.fusion.model import FusionClassifier  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402
from src.neurolm.config import load_config, save_json  # noqa: E402
from src.neurolm.splits import make_splits  # noqa: E402


def load_for_analysis(name, revision, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype,
                                              attn_implementation="eager")
    return lm.to(device=device, dtype=dtype), tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_text_lora.yaml")
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--lora-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--word-targets", default=None, help="word_targets.csv from analyze_eeg_signal.py")
    parser.add_argument("--arm", default="text_eeg")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=400)
    parser.add_argument("--model", default=None, help="must match the model used for training")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-perm", type=int, default=20)
    return parser.parse_args()


def plot_attention(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for role, part in summary.groupby("role"):
        axes[0].plot(part["layer"], part["mass"], label=role)
        axes[1].plot(part["layer"], part["per_token"], label=role)
    axes[0].set_title("attention mass from the answer position")
    axes[1].set_title("attention per token")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    cfg = load_config(args.config)
    manifest_path = os.path.join(args.lora_run_dir, "manifest.json")
    weights = os.path.join(args.lora_run_dir, args.arm, f"fold_{args.fold}_weights.pt")
    if not os.path.exists(manifest_path) or not os.path.exists(weights):
        raise SystemExit(f"no trained model at {weights}: run scripts/run_eeg_text_lora.py first "
                         "(notebook step 6) and let it finish at least this arm and fold")
    manifest = json.load(open(manifest_path))
    trained_model = manifest["settings"]["model"]
    model_name = args.model or trained_model
    revision = manifest["settings"].get("revision")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    trials = load_word_eeg(args.word_eeg_dir)
    data = build_fusion_data(trials, log_power=cfg["eeg"].get("log_power", "auto"))
    split_cfg = cfg["split"]
    split = make_splits(split_cfg["protocol"], data.samples, split_cfg["seed"], split_cfg["params"])[args.fold]
    stats, _ = reader_statistics(data, split.train)
    rng = np.random.default_rng(split_cfg["seed"] * 1000 + args.fold)
    aligned = model_inputs(data, stats, "aligned")
    variants = {
        "aligned": aligned,
        "shuffled_within_reader": model_inputs(data, stats, "shuffled", split, rng),
        "fixation_only": model_inputs(data, stats, "fixation_only"),
    }
    pick = np.random.default_rng(0)
    rows = np.sort(pick.choice(split.test, size=min(args.max_trials, len(split.test)), replace=False))
    variants["other_reader_same_sentence"] = fusion.other_reader_inputs(data, aligned, rows, pick)

    lm, tokenizer = load_for_analysis(model_name, revision, dtype, device)
    model = FusionClassifier(lm, tokenizer, class_words=cfg["class_words"], eeg_dim=data.dim,
                             prompt=cfg["prompt"], lora=cfg["lora"], projector=cfg["projector"]).to(device)
    model.load_trainable_state(torch.load(weights, map_location="cpu"))
    n_layers = len(model.decoder.layers) if hasattr(model.decoder, "layers") else lm.config.num_hidden_layers
    layers = sorted({n_layers // 4, n_layers // 2, n_layers})
    labels = data.samples["label_id"].to_numpy()[rows]
    print(f"{args.arm} fold {args.fold}: {len(rows)} held-out trials, layers {layers}, {model_name} ({dtype})")

    reliance = fusion.counterfactual_reliance(model, rows, data, variants, labels)
    reliance.to_csv(os.path.join(args.out_dir, "counterfactual_reliance.csv"), index=False)
    print(reliance.round(4).to_string(index=False))
    attention, slots = fusion.attention_and_states(model, rows, data, aligned, layers)
    attention.to_csv(os.path.join(args.out_dir, "attention_by_layer.csv"), index=False)
    plot_attention(attention, os.path.join(args.out_dir, "attention_by_layer.png"))
    attribution = fusion.gradient_attribution(model, rows, data, aligned)
    attribution.describe().to_csv(os.path.join(args.out_dir, "gradient_attribution_summary.csv"))
    summary = {"arm": args.arm, "fold": args.fold, "n_trials": int(len(rows)), "layers": layers,
               "attribution_mean_share": attribution.mean().to_dict(),
               "attention_mean_share_over_layers": attention.groupby("role")[["mass", "per_token"]].mean().to_dict(),
               "reliance": reliance.to_dict("records")}
    if args.word_targets:
        word_table = pd.read_csv(args.word_targets)
        probes = fusion.probe_slot_representations(slots, data, word_table, layers, n_perm=args.n_perm)
        probes.to_csv(os.path.join(args.out_dir, "slot_probes.csv"), index=False)
        print(probes.round(4).to_string(index=False))
        summary["slot_probes"] = probes.to_dict("records")
    summary["runtime_s"] = time.time() - started
    save_json(summary, os.path.join(args.out_dir, "stage3_summary.json"))
    print(f"done in {time.time() - started:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
