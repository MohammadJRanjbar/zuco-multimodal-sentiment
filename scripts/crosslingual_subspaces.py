"""Compare the EEG-predictive directions of English (ZuCo) and Persian (TeCo) in one multilingual text space.

1. Overlap of the two languages' EEG subspaces (LaBSE word vectors -> top-k EEG
   components, within-sentence centered), against a null from EEG targets
   shuffled across words, before and after removing word-feature directions
   (length, frequency, position) of both languages.
2. Transfer: held-out R^2 for one language's EEG from its word vectors
   projected onto the other language's EEG subspace, compared with the full
   space, random subspaces, the shuffled-EEG subspace and the other language's
   word-feature subspace.
Writes crosslingual.md/.json to --out-dir.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from src.brainshaping import scan_io  # noqa: E402
from src.brainshaping.data import fit_targets  # noqa: E402
from src.brainshaping.encoding import (  # noqa: E402
    ALPHAS, center_by_group, extract_layer_vectors, prepare_folds, scan_layer, summarize_layer, word_controls,
)
from src.followup import crosslingual as cl  # noqa: E402
from src.progress import progress  # noqa: E402

WORD_FEATURES = ["log_length", "zipf", "relative_position"]
NAMES = {"en": "English (ZuCo)", "fa": "Persian (TeCo)"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zuco-word-eeg-dir", required=True)
    parser.add_argument("--teco-trt-dir", required=True)
    parser.add_argument("--teco-labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vector-cache-dir", default=None)
    parser.add_argument("--model", default="labse")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_language(lang, args, device):
    dataset = "zuco" if lang == "en" else "teco"
    data = scan_io.load_items(dataset, args.zuco_word_eeg_dir, args.teco_trt_dir, args.teco_labels_csv)
    items = data["items"]
    item_sentence, item_word = items["sentence_row"].to_numpy(), items["word_index"].to_numpy()
    (alias, name), = scan_io.model_specs([args.model])
    cache = args.vector_cache_dir or os.path.join(args.out_dir, "vectors")
    vectors, found, _ = scan_io.word_vectors(cache, dataset, alias, name, data["sentences"], item_sentence, item_word,
                                             device, extractor=extract_layer_vectors)
    keep = np.flatnonzero(found)
    controls, _, _ = word_controls(data["sentences"]["words"].tolist(), items, data["meta"],
                                   data["readers_per_sentence"], lang)
    groups = items["sentence_id"].to_numpy()[keep]
    codes = np.unique(groups, return_inverse=True)[1]
    return {"X": center_by_group(np.array(vectors[args.layer])[keep].astype(np.float64), codes),
            "eeg_raw": data["eeg"][keep], "eeg": center_by_group(data["eeg"][keep].astype(np.float64), codes),
            "L": center_by_group(controls[WORD_FEATURES].to_numpy(dtype=np.float64)[keep], codes),
            "groups": groups}


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    rng = np.random.default_rng(args.seed)
    langs = {lang: load_language(lang, args, device) for lang in ("en", "fa")}
    scale = np.concatenate([langs[l]["X"] for l in langs]).std(axis=0)
    scale[scale < 1e-8] = 1.0
    for d in langs.values():
        d["Xs"] = d["X"] / scale
        transform, explained = fit_targets(d["eeg"], args.k)
        d["Y"], d["explained"] = transform(d["eeg"]).astype(np.float64), explained
        W, d["alpha"] = cl.ridge_coefficients(d["Xs"], d["Y"], d["groups"], args.inner_folds, device=device)
        d["U"] = cl.orth(W)
        Lz = (d["L"] - d["L"].mean(0)) / np.where(d["L"].std(0) > 0, d["L"].std(0), 1)
        W_lex, _ = cl.ridge_coefficients(d["Xs"], Lz, d["groups"], args.inner_folds, device=device)
        d["U_lex"] = cl.orth(W_lex)
        d["U_null"] = []
        for _ in progress(range(args.n_null), desc="null subspaces", unit="fit"):
            W_null, _ = cl.ridge_coefficients(d["Xs"], d["Y"][rng.permutation(len(d["Y"]))], d["groups"],
                                              args.inner_folds, device=device)
            d["U_null"].append(cl.orth(W_null))
    en, fa = langs["en"], langs["fa"]
    P = np.hstack([en["U_lex"], fa["U_lex"]])
    dim = en["Xs"].shape[1]
    real = cl.overlap(en["U"], fa["U"])
    null = [cl.overlap(en["U"], u) for u in fa["U_null"]] + [cl.overlap(u, fa["U"]) for u in en["U_null"]]
    real_res = cl.overlap(cl.remove_subspace(en["U"], P), cl.remove_subspace(fa["U"], P))
    null_res = ([cl.overlap(cl.remove_subspace(en["U"], P), cl.remove_subspace(u, P)) for u in fa["U_null"]]
                + [cl.overlap(cl.remove_subspace(u, P), cl.remove_subspace(fa["U"], P)) for u in en["U_null"]])
    summary = {"model": args.model, "layer": args.layer, "k": args.k, "dim": dim,
               "explained": {l: langs[l]["explained"] for l in langs},
               "overlap": {"real": real, "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95)),
                           "random_expectation": args.k / dim,
                           "real_without_word_features": real_res, "null_without_word_features_mean": float(np.mean(null_res)),
                           "null_without_word_features_q95": float(np.quantile(null_res, 0.95))},
               "eeg_subspace_contains_word_features": {l: cl.overlap(langs[l]["U_lex"], langs[l]["U"]) for l in langs},
               "transfer": {}}
    for target, source in (("fa", "en"), ("en", "fa")):
        t, s = langs[target], langs[source]
        prepared = prepare_folds(t["eeg_raw"], t["groups"], ("centered",), args.k, args.folds, args.inner_folds,
                                 args.seed)
        folds, codes = prepared["modes"]["centered"], prepared["codes"]
        random_bases = [cl.orth(rng.standard_normal((dim, args.k))) for _ in range(5)]
        other = {"en": "English", "fa": "Persian"}[source]
        features = {"full text space": t["Xs"],
                    f"{other} EEG subspace": t["Xs"] @ s["U"],
                    f"{other} EEG subspace without word features": t["Xs"] @ cl.remove_subspace(s["U"], P),
                    f"{other} shuffled-EEG subspace": t["Xs"] @ s["U_null"][0],
                    f"{other} word-feature subspace": t["Xs"] @ s["U_lex"]}
        results = {}
        for name, Z in progress(features.items(), desc=f"transfer to {target}", unit="feature set"):
            results[name] = summarize_layer(scan_layer(Z, folds, codes, "centered", ALPHAS, device),
                                            args.n_boot, args.seed)
        random = [summarize_layer(scan_layer(t["Xs"] @ B, folds, codes, "centered", ALPHAS, device),
                                  args.n_boot, args.seed)["r2"] for B in random_bases]
        summary["transfer"][target] = {
            **{name: {k: r[k] for k in ("r2", "r2_ci_low", "r2_ci_high", "r2_shuffled")} for name, r in results.items()},
            "random subspace (mean of 5)": {"r2": float(np.mean(random))}}
    summary["runtime_min"] = (time.time() - started) / 60
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out_dir, "crosslingual.json"), "w"), indent=1)
    write_report(os.path.join(args.out_dir, "crosslingual.md"), summary)
    print(open(os.path.join(args.out_dir, "crosslingual.md")).read())


def write_report(path, s):
    o = s["overlap"]
    lines = ["# English vs Persian EEG directions in a shared text space", "",
             f"{s['model']} layer {s['layer']} ({s['dim']} dimensions), {s['k']} EEG components per language, "
             "within-sentence centered.", "",
             "## Overlap of the two EEG subspaces", "",
             "Mean squared cosine of the principal angles (0 = unrelated, 1 = identical).", "",
             "| | real | null mean | null 95th percentile |", "|---|---:|---:|---:|",
             f"| all directions | {o['real']:.4f} | {o['null_mean']:.4f} | {o['null_q95']:.4f} |",
             f"| without word-feature directions | {o['real_without_word_features']:.4f} | "
             f"{o['null_without_word_features_mean']:.4f} | {o['null_without_word_features_q95']:.4f} |", "",
             f"Random subspaces would give about {o['random_expectation']:.4f}. Share of each language's word-feature "
             "directions inside its own EEG subspace: "
             + ", ".join(f"{NAMES[l]} {v:.2f}" for l, v in s["eeg_subspace_contains_word_features"].items()) + ".", "",
             "## Transfer: predicting one language's EEG through the other's directions", ""]
    for target, results in s["transfer"].items():
        lines += [f"**{NAMES[target]}** (held-out R², within-sentence)", "", "| features | R² [95% CI] |", "|---|---|"]
        for name, r in results.items():
            ci = f" [{r['r2_ci_low']:.4f}, {r['r2_ci_high']:.4f}]" if "r2_ci_low" in r else ""
            lines.append(f"| {name} | {r['r2']:.4f}{ci} |")
        lines.append("")
    shared = o["real"] > o["null_q95"]
    beyond = o["real_without_word_features"] > o["null_without_word_features_q95"]
    lines += ["## Reading", "",
              "* **Shared EEG directions:** " + ("the overlap exceeds the shuffled-EEG null." if shared else
                                                 "the overlap is within the shuffled-EEG null."),
              "* **Beyond word features:** " + ("the overlap remains after removing length, frequency and position."
                                                if beyond else "no overlap remains once length, frequency and position "
                                                "directions are removed.")]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
