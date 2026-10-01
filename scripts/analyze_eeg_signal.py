"""Stages 1-2: what does word-level EEG encode, and which representation keeps it?

Needs only the word-EEG cache (scripts/extract_word_eeg.py) and the labels CSV;
runs on CPU. Optional extras for stage 2: the NeuroLM embedding cache and the
handcrafted feature cache.
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

from src.diagnostics import signal  # noqa: E402
from src.diagnostics.representations import build_representations, probe_representations  # noqa: E402
from src.diagnostics.targets import build_word_table, lm_surprisal, sentence_targets, valence_lexicon  # noqa: E402
from src.diagnostics.topomap import plot_topomaps  # noqa: E402
from src.fusion.word_eeg import BANDS, load_word_eeg  # noqa: E402
from src.neurolm.config import save_json  # noqa: E402

WORD_TARGETS = {
    "length": "regression", "zipf": "regression", "surprisal": "regression", "trt_ms": "regression",
    "is_content": "binary", "relative_position": "regression",
    "valence": "regression", "abs_valence": "regression", "in_lexicon": "binary",
    "sentence_label": "multiclass",
}
POSITIVE_CONTROLS = ["length", "zipf", "surprisal", "trt_ms", "is_content", "relative_position"]
SENTIMENT_TARGETS = ["valence", "abs_valence", "in_lexicon", "sentence_label"]
COVARIATES = ["length", "zipf", "surprisal", "relative_position", "is_content"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--surprisal-model", default="gpt2", help="'none' to skip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-perm", type=int, default=50, help="permutations for reader-averaged probes")
    parser.add_argument("--n-perm-single", type=int, default=20, help="permutations for single-reader probes")
    parser.add_argument("--neurolm-cache-dir", default=None)
    parser.add_argument("--neurolm-config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--handcrafted-dir", default=None)
    return parser.parse_args()


def probe_rows(X, targets, groups, units, n_perm, level, sentence_units):
    rows = []
    for target, task in WORD_TARGETS.items():
        if target not in targets or targets[target].isna().all():
            continue
        values = targets[target].to_numpy()
        keep = ~pd.isna(values)
        y = values[keep].astype(float if task == "regression" else int)
        if task != "regression" and len(np.unique(y)) < 2:
            continue
        perm_units = (sentence_units if target == "sentence_label" else units)[keep]
        started = time.time()
        result = signal.ridge_probe(X[keep], y, groups[keep], task=task, n_perm=n_perm, perm_units=perm_units)
        if result.get("skipped"):
            continue
        kind = "positive control" if target in POSITIVE_CONTROLS else "sentiment"
        rows.append({"level": level, "target": target, "kind": kind, "metric": result["metric"], "n": result["n"],
                     "n_perm": n_perm,
                     "score": result["score"], "null_mean": result.get("null_mean"),
                     "null_q95": result.get("null_q95"), "p_value": result.get("p_value")})
        print(f"  [{level}] {target:>18s}: {result['metric']} {result['score']:.4f} "
              f"(null q95 {result.get('null_q95', float('nan')):.4f}, p={result.get('p_value')}) {time.time() - started:.0f}s")
    return rows


def main():
    args = parse_args()
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    started = time.time()
    trials = load_word_eeg(args.word_eeg_dir)
    print(f"{len(trials)} reader x sentence trials")

    lexicon = valence_lexicon()
    if not lexicon:
        print("WARNING: vaderSentiment not installed; valence targets will be missing")
    surprisal = None
    if args.surprisal_model.lower() != "none":
        unique = {}
        for trial in trials:
            unique.setdefault(trial["sentence_id"], trial["words"])
        ids = sorted(unique)
        values = lm_surprisal([unique[i] for i in ids], args.surprisal_model, args.device)
        surprisal = dict(zip(ids, values))
    word_table = build_word_table(trials, lexicon=lexicon, surprisal=surprisal)
    word_table.to_csv(os.path.join(args.out_dir, "word_targets.csv"), index=False)

    meta, X, info = signal.long_word_table(trials)
    print(f"{len(meta)} word observations with EEG, {X.shape[1]} features, log={info['log_transformed']}")
    variance = signal.variance_components(meta, X)
    variance.to_csv(os.path.join(args.out_dir, "variance_components.csv"), index=False)
    Z = signal.zscore_per_reader(meta, X)
    del X  # keep one copy of the word EEG in memory
    reliability, reliability_info = signal.split_half_reliability(meta, Z)
    reliability.to_csv(os.path.join(args.out_dir, "reliability.csv"), index=False)
    n_channels = info["n_channels"]
    plot_topomaps(reliability["reliability_all_readers"].to_numpy().reshape(len(BANDS), n_channels),
                  [f"{band}" for band in BANDS], os.path.join(plots, "reliability_all_readers.png"),
                  "Split-half reliability of reader-averaged word EEG", cmap="viridis")

    item_meta, A = signal.reader_average(meta, Z)
    single_targets = meta.merge(word_table, on=["sentence_id", "word_index"], how="left")
    item_targets = item_meta.merge(word_table, on=["sentence_id", "word_index"], how="left")
    print("word-level probes (single reader):")
    rows = probe_rows(Z, single_targets, meta["sentence_id"].to_numpy(), meta["item"].to_numpy(),
                      args.n_perm_single, "single reader", meta["sentence_id"].to_numpy())
    print("word-level probes (reader-averaged):")
    rows += probe_rows(A, item_targets, item_meta["sentence_id"].to_numpy(), item_meta["item"].to_numpy(),
                       args.n_perm, "reader-averaged", item_meta["sentence_id"].to_numpy())
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "word_probes.csv"), index=False)

    band_rows = []
    for band, columns in signal.band_slices(n_channels).items():
        for target in ["surprisal", "zipf", "length", "trt_ms", "valence", "abs_valence"]:
            if target in item_targets and item_targets[target].notna().any():
                result = signal.ridge_probe(A[:, columns], item_targets[target].to_numpy(),
                                            item_meta["sentence_id"].to_numpy(), "regression")
                band_rows.append({"band": band, "target": target, "R2": result["score"]})
    pd.DataFrame(band_rows).to_csv(os.path.join(args.out_dir, "band_probes.csv"), index=False)

    for target in ["surprisal", "zipf", "trt_ms", "valence", "abs_valence"]:
        if target in item_targets and item_targets[target].notna().sum() > 50:
            maps = signal.feature_correlations(A, item_targets[target].to_numpy(dtype=float))
            plot_topomaps(maps, BANDS, os.path.join(plots, f"correlation_{target}.png"),
                          f"Correlation of reader-averaged word EEG with {target}")

    covariate_rows = []
    lexical = item_targets[item_targets["valence"].notna()] if "valence" in item_targets else item_targets.iloc[:0]
    if len(lexical) > 100:
        covariates = [c for c in COVARIATES if c in lexical and lexical[c].notna().all()]
        index = lexical.index.to_numpy()
        groups = lexical["sentence_id"].to_numpy()
        for name, design in (("covariates only", lexical[covariates].to_numpy(float)),
                             ("covariates + EEG", np.hstack([lexical[covariates].to_numpy(float), A[index]])),
                             ("EEG only", A[index])):
            result = signal.ridge_probe(design, lexical["valence"].to_numpy(float), groups, "regression")
            covariate_rows.append({"model": name, "target": "valence (lexicon words)", "R2": result["score"]})
    pd.DataFrame(covariate_rows).to_csv(os.path.join(args.out_dir, "valence_beyond_covariates.csv"), index=False)

    extra = {}
    if args.neurolm_cache_dir:
        from src.neurolm.config import load_config
        from src.neurolm.experiment import load_neurolm_view

        config = load_config(args.neurolm_config)
        _, metadata, features, _ = load_neurolm_view(config, config["probe"]["primary_view"], args.neurolm_cache_dir)
        frame = pd.DataFrame({"reader": metadata["subject_id"], "sentence_id": metadata["sentence_id"]})
        for key in ("tokenizer__mean", "gpt__mean"):
            extra[f"NeuroLM {key}"] = (frame, features[key])
    if args.handcrafted_dir:
        from src.neurolm.dataset import load_handcrafted_trials

        table, hand = load_handcrafted_trials(args.handcrafted_dir)
        extra["handcrafted sentence features"] = (pd.DataFrame({"reader": table["subject_id"],
                                                                 "sentence_id": table["sentence_id"]}), hand)
    representations = build_representations(meta, Z, extra)
    stage2 = probe_representations(representations, sentence_targets(word_table), n_perm=args.n_perm)
    stage2.to_csv(os.path.join(args.out_dir, "stage2_representations.csv"), index=False)
    print(stage2.round(4).to_string(index=False))

    band_reliability = {band: float(np.median(reliability["reliability_all_readers"].to_numpy()[columns]))
                        for band, columns in signal.band_slices(n_channels).items()}
    save_json({
        "n_trials": len(trials), "n_word_observations": int(len(meta)), "n_items": int(len(item_meta)),
        "feature_info": info, "lexicon_size": len(lexicon), "surprisal_model": args.surprisal_model,
        "variance_components_median": variance.median().to_dict(),
        "reliability": {**reliability_info,
                        "median_all_readers": float(reliability["reliability_all_readers"].median()),
                        "q90_all_readers": float(reliability["reliability_all_readers"].quantile(0.9)),
                        "median_single_reader": float(reliability["reliability_single_reader"].median()),
                        "median_all_readers_by_band": band_reliability},
        "runtime_s": time.time() - started,
    }, os.path.join(args.out_dir, "stage1_summary.json"))
    print(f"done in {time.time() - started:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
