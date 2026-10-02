"""Do text-model word vectors predict word-level EEG on unseen sentences?

Scans models x layers on ZuCo (English) or TeCo (Persian): ridge regression
from each layer's word vectors to the top-k principal components of
reader-averaged word EEG, with folds split by sentence (outer and inner) and a
shuffled-target control. Version A and brain-tuning can only help if some
layer passes here.

Writes, under --out-dir/encoding_scan_<dataset>/: one CSV per model (all
layers), encoding_scan_<dataset>.md (verdict + best layer per model) and a
plot. Models already scanned with the same settings are reused on re-runs.
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

from src.brainshaping.data import sentence_table  # noqa: E402
from src.brainshaping.encoding import (  # noqa: E402
    ALPHAS, _progress, extract_layer_vectors, noise_ceiling, prepare_folds, residualize_folds, scan_layer,
    summarize_layer, word_controls,
)
from src.diagnostics import signal  # noqa: E402

MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "me5-large": "intfloat/multilingual-e5-large",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
REVISIONS = {"Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["zuco", "teco"])
    parser.add_argument("--word-eeg-dir", help="ZuCo word-level EEG cache (extract_word_eeg.py)")
    parser.add_argument("--trt-dir", help="TeCo folder with <Name>_trt_total.pickle files")
    parser.add_argument("--labels-csv", help="TeCo teco_sentiment_labels_task1.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vector-cache-dir", default=None, help="word-vector cache (default: <out-dir>/vectors)")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help=f"aliases ({', '.join(MODELS)}), alias=name_or_path, or HF names")
    parser.add_argument("--modes", nargs="+", default=["centered", "raw"], choices=["centered", "raw"])
    parser.add_argument("--k", type=int, default=32, help="EEG principal components used as targets")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--layer-stride", type=int, default=1, help="scan every n-th layer (last always included)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-controls", action="store_true",
                        help="skip the lexical / reading-behaviour controls and the noise ceiling")
    return parser.parse_args()


def model_specs(entries):
    specs = []
    for entry in entries:
        if "=" in entry:
            alias, name = entry.split("=", 1)
        elif entry in MODELS:
            alias, name = entry, MODELS[entry]
        else:
            alias, name = entry.rstrip("/").split("/")[-1].lower(), entry
        specs.append((alias, name))
    return specs


def load_items(args):
    if args.dataset == "zuco":
        from src.fusion.word_eeg import load_word_eeg

        if not args.word_eeg_dir:
            raise SystemExit("--word-eeg-dir is required for ZuCo")
        trials, drop = load_word_eeg(args.word_eeg_dir), None
    else:
        from src.brainshaping.teco import load_teco_trials

        if not args.trt_dir:
            raise SystemExit("--trt-dir is required for TeCo")
        trials, drop = load_teco_trials(args.trt_dir, args.labels_csv), ()
    sentences = sentence_table(trials)
    # same steps as item_eeg, keeping the per-reader table for the controls and the noise ceiling
    meta, X, _ = (signal.long_word_table(trials) if drop is None
                  else signal.long_word_table(trials, drop_channels=drop))
    Z = signal.zscore_per_reader(meta, X)
    del X
    items, eeg = signal.reader_average(meta, Z)
    readers = pd.DataFrame({"sentence_id": [t["sentence_id"] for t in trials],
                            "reader": [t["subject_id"] for t in trials]})
    readers_per_sentence = readers.drop_duplicates().groupby("sentence_id").size().to_dict()
    return {"sentences": sentences, "items": items.reset_index(drop=True), "eeg": eeg, "meta": meta, "Z": Z,
            "readers_per_sentence": readers_per_sentence}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def word_vectors(args, alias, name, sentences, item_sentence, item_word, device):
    """Load cached vectors for (dataset, model) or extract them; returns a memory-mapped array."""
    cache_dir = args.vector_cache_dir or os.path.join(args.out_dir, "vectors")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.join(cache_dir, f"{args.dataset}_{alias}")
    revision = REVISIONS.get(name)
    key = digest([name, revision, sentences["words"].tolist(), item_sentence.tolist(), item_word.tolist()])
    if os.path.exists(stem + ".json") and os.path.exists(stem + ".npy"):
        meta = json.load(open(stem + ".json"))
        if meta["key"] == key:
            print(f"{alias}: reusing cached word vectors")
            return np.load(stem + ".npy", mmap_mode="r"), np.array(meta["found"], dtype=bool), meta["info"]
    print(f"{alias}: extracting word vectors from every layer of {name}")
    vectors, found, info = extract_layer_vectors(sentences["words"].tolist(), item_sentence, item_word, name, device,
                                                 batch_size=args.batch_size, revision=revision,
                                                 desc=f"{alias}: word vectors")
    np.save(stem + ".npy", vectors)
    json.dump({"key": key, "found": found.tolist(), "info": info}, open(stem + ".json", "w"))
    del vectors
    return np.load(stem + ".npy", mmap_mode="r"), found, info


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    data = load_items(args)
    sentences, items, eeg = data["sentences"], data["items"], data["eeg"]
    row_of = {sid: i for i, sid in enumerate(sentences["sentence_id"])}
    item_sentence = items["sentence_id"].map(row_of).to_numpy()
    item_word = items["word_index"].to_numpy()
    groups = items["sentence_id"].to_numpy()
    print(f"{args.dataset}: {len(sentences)} sentences, {len(items)} words with EEG, "
          f"{eeg.shape[1]} EEG features per word (reader-averaged); ridge on {device}")

    scan_dir = os.path.join(args.out_dir, f"encoding_scan_{args.dataset}")
    os.makedirs(scan_dir, exist_ok=True)
    settings = {"dataset": args.dataset, "n_items": int(len(items)), "k": args.k, "folds": args.folds,
                "inner_folds": args.inner_folds, "modes": args.modes, "alphas": ALPHAS.tolist(),
                "layer_stride": args.layer_stride, "n_boot": args.n_boot, "seed": args.seed,
                "eeg_digest": hashlib.sha256(np.ascontiguousarray(eeg).tobytes()).hexdigest()[:16]}
    key = digest(settings)
    everything = np.ones(len(items), dtype=bool)
    prepared_by_mask = {hashlib.sha256(everything.tobytes()).hexdigest():
                        prepare_folds(eeg, groups, args.modes, args.k, args.folds, args.inner_folds, args.seed)}
    explained = {mode: float(np.mean([f["explained"] for f in folds]))
                 for mode, folds in next(iter(prepared_by_mask.values()))["modes"].items()}
    ceilings = {}
    if not args.no_controls:
        for mode in args.modes:
            reliability, _ = noise_ceiling(data["meta"], data["Z"], eeg, groups, args.k,
                                           centered=mode == "centered", seed=args.seed)
            if reliability is not None:
                ceilings[mode] = reliability
    del data["Z"]
    frames = []
    for alias, name in model_specs(args.models):
        csv_path = os.path.join(scan_dir, f"{alias}.csv")
        if os.path.exists(csv_path):
            previous = pd.read_csv(csv_path)
            if len(previous) and (previous["settings_key"] == key).all() and (previous["model_name"] == name).all():
                print(f"{alias}: reusing finished scan ({csv_path})")
                frames.append(previous)
                continue
        vectors, found, info = word_vectors(args, alias, name, sentences, item_sentence, item_word, device)
        if not found.all():
            print(f"{alias}: {int((~found).sum())} words produced no token and are left out")
        mask_key = hashlib.sha256(found.tobytes()).hexdigest()
        if mask_key not in prepared_by_mask:
            prepared_by_mask[mask_key] = prepare_folds(eeg[found], groups[found], args.modes, args.k,
                                                       args.folds, args.inner_folds, args.seed)
        prepared = prepared_by_mask[mask_key]
        n_layers = vectors.shape[0]
        layers = sorted(set(range(0, n_layers, args.layer_stride)) | {n_layers - 1})
        rows, best = [], {}
        bar = _progress(total=len(layers) * len(args.modes) * args.folds, desc=f"{alias}: ridge scan", unit="fold")
        for layer in layers:
            X = np.asarray(vectors[layer])[found]
            for mode in args.modes:
                result = scan_layer(X, prepared["modes"][mode], prepared["codes"], mode, ALPHAS, device,
                                    on_fold=bar.update)
                summary = summarize_layer(result, args.n_boot, args.seed)
                rows.append({"model": alias, "model_name": name, "pool": info["pool"], "layer": layer,
                             "n_layers": n_layers - 1, "depth": layer / max(n_layers - 1, 1), "mode": mode,
                             "n_items": int(found.sum()), **summary,
                             "fold_r2": json.dumps(summary["fold_r2"]),
                             "component_r2": json.dumps(summary["component_r2"]), "settings_key": key})
                best[mode] = max(best.get(mode, -np.inf), summary["r2"])
            bar.set_postfix({f"best R2 {m}": f"{v:.3f}" for m, v in best.items()}, refresh=False)
        bar.close()
        frame = pd.DataFrame(rows)
        frame.to_csv(csv_path, index=False)
        frames.append(frame)
        del vectors

    results = pd.concat(frames, ignore_index=True)
    controls = None
    if not args.no_controls and "centered" in args.modes:
        controls = run_controls(args, results, next(iter(prepared_by_mask.values())), data, item_sentence, item_word,
                                device)
        controls["table"].to_csv(os.path.join(scan_dir, f"controls_{args.dataset}.csv"), index=False)
        json.dump({"baselines": controls["baselines"], "lexical": controls["lexical"],
                   "reading": controls["reading"], "ceiling": {m: v.tolist() for m, v in ceilings.items()}},
                  open(os.path.join(scan_dir, f"controls_{args.dataset}.json"), "w"), indent=1, default=float)
    info = {"dataset": args.dataset, "n_sentences": int(len(sentences)), "n_items": int(len(items)),
            "n_features": int(eeg.shape[1]), "k": args.k, "folds": args.folds, "explained": explained,
            "runtime_s": time.time() - started}
    report = os.path.join(scan_dir, f"encoding_scan_{args.dataset}.md")
    write_report(report, results, info, controls, ceilings)
    plot(os.path.join(scan_dir, f"encoding_scan_{args.dataset}.png"), results, args.dataset)
    print(open(report).read())


def run_controls(args, results, prepared, data, item_sentence, item_word, device):
    """Best centered layer per model: is its EEG prediction more than lexical / reading features?"""
    table, lexical, reading = word_controls(data["sentences"]["words"].tolist(),
                                            data["items"].assign(sentence_row=item_sentence), data["meta"],
                                            data["readers_per_sentence"], "en" if args.dataset == "zuco" else "fa")
    usable = [c for c in reading if np.isfinite(table[c]).all()]
    if len(usable) < len(reading):
        print(f"controls: reading features with missing values left out: {sorted(set(reading) - set(usable))}")
    sets = {"lexical": lexical, "lexical+reading": lexical + usable}
    folds, codes = prepared["modes"]["centered"], prepared["codes"]
    centered = results[results["mode"] == "centered"]
    best = centered.loc[centered.groupby("model", sort=False)["r2"].idxmax()]
    bar = _progress(total=(len(sets) + 3 * len(best)) * args.folds, desc="controls", unit="fold")
    baselines, residual = {}, {}
    for name, columns in sets.items():
        L = table[columns].to_numpy(dtype=np.float64)
        summary = summarize_layer(scan_layer(L, folds, codes, "centered", ALPHAS, device, on_fold=bar.update),
                                  args.n_boot, args.seed)
        baselines[name] = {k: v for k, v in summary.items() if k != "fold_r2"}
        residual[name] = residualize_folds(folds, L, codes, centered=True)
    rows = []
    for _, top in best.iterrows():
        vectors, found, _ = word_vectors(args, top["model"], top["model_name"], data["sentences"], item_sentence,
                                         item_word, device)
        if not found.all():
            print(f"controls: {top['model']} skipped ({int((~found).sum())} words without a vector)")
            bar.update(3 * args.folds)
            continue
        X = np.array(vectors[int(top["layer"])])
        full = summarize_layer(scan_layer(X, folds, codes, "centered", ALPHAS, device, on_fold=bar.update),
                               args.n_boot, args.seed)
        row = {"model": top["model"], "layer": int(top["layer"]), "n_layers": int(top["n_layers"]),
               "r2_vectors": full["r2"], "r2_vectors_ci_low": full["r2_ci_low"],
               "r2_vectors_ci_high": full["r2_ci_high"], "component_r2": json.dumps(full["component_r2"])}
        for name in sets:
            summary = summarize_layer(scan_layer(X, residual[name], codes, "centered", ALPHAS, device,
                                                 on_fold=bar.update), args.n_boot, args.seed)
            for key in ("r2", "r2_ci_low", "r2_ci_high", "r2_shuffled", "delta_ci_low"):
                row[f"beyond_{name}_{key}"] = summary[key]
        rows.append(row)
        del vectors
    bar.close()
    return {"table": pd.DataFrame(rows), "baselines": baselines, "lexical": lexical, "reading": usable}


def controls_section(controls, ceilings):
    lines = ["## Is it more than word length, frequency and reading behaviour? (centered)", ""]
    ceiling = ceilings.get("centered")
    if ceiling is not None:
        lines.append(f"**Noise ceiling:** {100 * np.clip(ceiling, 0, None).mean():.1f}% of the word-to-word EEG "
                     "variance repeats across readers (split-half reliability of the reader average, mean over the "
                     f"{len(ceiling)} components; best component {100 * ceiling.max():.1f}%). No model can explain "
                     "more than this.")
        lines.append("")
    for name, values in controls["baselines"].items():
        columns = controls["lexical"] + (controls["reading"] if name == "lexical+reading" else [])
        lines.append(f"* **{name} features alone** ({', '.join(columns)}): R² {values['r2']:.4f} "
                     f"[{values['r2_ci_low']:.4f}, {values['r2_ci_high']:.4f}]")
    lines += ["", "| model | layer | text vectors R² | beyond lexical R² [95% CI] | beyond lexical + reading R² "
              "[95% CI] | share of ceiling |", "|---|---:|---|---|---|---:|"]
    table = controls["table"]
    for _, row in table.iterrows():
        share = (f"{100 * row['r2_vectors'] / np.clip(ceiling, 0, None).mean():.0f}%"
                 if ceiling is not None and np.clip(ceiling, 0, None).mean() > 0 else "—")
        cells = [f"{row[f'beyond_{n}_r2']:.4f} [{row[f'beyond_{n}_r2_ci_low']:.4f}, {row[f'beyond_{n}_r2_ci_high']:.4f}]"
                 for n in ("lexical", "lexical+reading")]
        lines.append(f"| {row['model']} | {row['layer']}/{row['n_layers']} | {row['r2_vectors']:.4f} "
                     f"[{row['r2_vectors_ci_low']:.4f}, {row['r2_vectors_ci_high']:.4f}] | {cells[0]} | {cells[1]} | "
                     f"{share} |")
    lines.append("")
    for name in ("lexical", "lexical+reading"):
        survives = table[(table[f"beyond_{name}_r2_ci_low"] > 0) & (table[f"beyond_{name}_delta_ci_low"] > 0)]
        if len(survives):
            lines.append(f"* **Beyond {name}: signal remains** in {', '.join(survives['model'])} — the text vectors "
                         f"predict EEG that {name} features do not.")
        else:
            lines.append(f"* **Beyond {name}: nothing remains** — what the text vectors predict about EEG is "
                         f"accounted for by {name} features.")
    if len(table):
        top = table.loc[table["r2_vectors"].idxmax()]
        components = np.array(json.loads(top["component_r2"]))
        order = np.argsort(components)[::-1][:5]
        lines += ["", f"R² per EEG component ({top['model']} layer {top['layer']}; component 1 = largest EEG "
                  "variance): " + ", ".join(f"#{i + 1} {components[i]:.4f}" for i in order)
                  + f"; {int((components > 0.005).sum())} of {len(components)} components above 0.005.",
                  "", "*Beyond* = R² for EEG with the control features' (training-set) linear prediction removed, "
                  "as a share of what remains."]
    return lines


def passes(frame):
    return (frame["r2_ci_low"] > 0) & (frame["delta_ci_low"] > 0)


def write_report(path, results, info, controls=None, ceilings=None):
    n_candidates = len(results.groupby(["model", "layer"]))
    explained = ", ".join(f"{mode} {100 * v:.0f}%" for mode, v in info["explained"].items())
    lines = [f"# Do word vectors predict word-level EEG? ({info['dataset']})", "",
             f"{info['n_sentences']} sentences, {info['n_items']} words with EEG ({info['n_features']} EEG features "
             f"per word, averaged over readers). Targets: top {info['k']} EEG principal components "
             f"(share of EEG variance: {explained}). {info['folds']} folds split by sentence; the ridge penalty "
             "is chosen inside the training sentences only.", "",
             "**How to read:** R² is measured on sentences the model never saw. Above 0 means the word vectors "
             "predict EEG better than the average does. `shuffled` is the same pipeline with EEG targets shuffled "
             "across words (should be about 0). *centered*: only word-to-word differences within a sentence; "
             "*raw*: also differences between sentences. A layer **passes** when both its R² and its margin over "
             "shuffled have 95% CIs above 0.", "", "## Verdict", ""]
    for mode, frame in results.groupby("mode", sort=False):
        top = frame.loc[frame["r2"].idxmax()]
        n_pass = int(passes(frame).sum())
        where = (f"{top['model']} layer {int(top['layer'])}/{int(top['n_layers'])}: R² {top['r2']:.4f} "
                 f"[{top['r2_ci_low']:.4f}, {top['r2_ci_high']:.4f}], above shuffled by {top['delta']:.4f} "
                 f"[{top['delta_ci_low']:.4f}, {top['delta_ci_high']:.4f}]")
        verdict = "PASS" if passes(frame.loc[[top.name]]).iloc[0] else "FAIL"
        lines.append(f"* **{mode}: {verdict}** — best {where}; {n_pass} of {len(frame)} model-layers pass.")
    lines += ["", f"> The best layer is the maximum over {n_candidates} model-layers, so a single layer barely "
              "above 0 is weak evidence. Convincing: several neighbouring layers pass, and the same models and "
              "depths also pass on the other language.", "", "## Best layer per model", "",
              "| model | mode | best layer | R² [95% CI] | shuffled R² | R² − shuffled [95% CI] | layers passing | "
              "penalty at grid max |", "|---|---|---:|---|---:|---|---:|---:|"]
    for (model, mode), frame in results.groupby(["model", "mode"], sort=False):
        top = frame.loc[frame["r2"].idxmax()]
        lines.append(f"| {model} | {mode} | {int(top['layer'])}/{int(top['n_layers'])} | {top['r2']:.4f} "
                     f"[{top['r2_ci_low']:.4f}, {top['r2_ci_high']:.4f}] | {top['r2_shuffled']:.4f} | "
                     f"{top['delta']:.4f} [{top['delta_ci_low']:.4f}, {top['delta_ci_high']:.4f}] | "
                     f"{int(passes(frame).sum())}/{len(frame)} | {100 * frame['share_alpha_at_max'].mean():.0f}% |")
    lines += ["", "*Penalty at grid max*: share of folds where the strongest penalty won, i.e. the model chose to "
              "predict almost nothing (expected when there is no signal).", "",
              f"All layers: one CSV per model next to this report; plot `encoding_scan_{info['dataset']}.png`. "
              f"Runtime {info['runtime_s'] / 60:.1f} min."]
    if controls is not None:
        lines += [""] + controls_section(controls, ceilings or {})
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot(path, results, dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(results["mode"]))
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.2), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for i, (model, frame) in enumerate(results[results["mode"] == mode].groupby("model", sort=False)):
            frame = frame.sort_values("depth")
            color = f"C{i}"
            ax.plot(frame["depth"], frame["r2"], color=color, label=model)
            ax.fill_between(frame["depth"], frame["r2_ci_low"], frame["r2_ci_high"], color=color, alpha=0.2)
            ax.plot(frame["depth"], frame["r2_shuffled"], color=color, ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{dataset}: {mode} (dashed = shuffled EEG)")
        ax.set_xlabel("layer depth (0 = embeddings, 1 = last layer)")
        ax.set_ylabel("R² on held-out sentences")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
