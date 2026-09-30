"""Mandatory sanity checks for the frozen EEG-only probe.

A. label permutation: sentence-level null (primary) and trial-level null;
B. shuffled sentence association (EEG moved to other sentences of the same
   subject inside each split partition);
C. subject-ID decoding from the same embeddings (unseen-sentence split);
D. Gaussian random features of the same dimensionality;
E. label-distribution baselines;
F. embedding collapse diagnostics;
plus a duration-only baseline (trial length could proxy sentence identity).

Every permutation re-runs the whole pipeline, including hyperparameter
selection. To keep the cost bounded, permutation k uses one seed (rotating
through the configured seeds); a single-seed null is wider than the
seed-averaged observed statistic, which makes the p-value conservative.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neurolm import sanity  # noqa: E402
from src.neurolm.config import load_config, run_manifest, save_json, set_path_overrides  # noqa: E402
from src.neurolm.evaluation import run_probe  # noqa: E402
from src.neurolm.experiment import build_probe_data, feature_matrix, save_probe_result  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--cache-dir")
    parser.add_argument("--handcrafted-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--run-tag", default="probe_v1")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--protocols", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true", help="3 permutations, joint protocol, one seed")
    return parser.parse_args()


def main():
    args = parse_args()
    config = set_path_overrides(load_config(args.config), cache_dir=args.cache_dir,
                                handcrafted_dir=args.handcrafted_dir, results_dir=args.results_dir)
    probe, checks = config["probe"], config["sanity"]
    seeds = probe["seeds"]
    protocols = args.protocols or checks["permutation_protocols"]
    n_perm = args.n_permutations or checks["n_label_permutations"]
    n_trial_perm = checks["trial_level_permutations"]
    if args.smoke:
        protocols, seeds, n_perm, n_trial_perm = ["joint"], seeds[:1], 3, 2
    run_dir = os.path.join(config["paths"]["results_dir"], args.run_tag)
    out = os.path.join(run_dir, "sanity")
    os.makedirs(out, exist_ok=True)
    started = time.time()

    data = build_probe_data(config)
    samples = data["samples"].copy()
    X = feature_matrix(data["neurolm"], probe["primary_feature"])
    clf = checks["permutation_classifier"]
    provenance = {"view": probe["primary_view"], "feature": probe["primary_feature"],
                  "checkpoint": data["payload"]["checkpoint"], "channel_mapping": data["payload"]["channels"]}
    save_json(run_manifest(config, {**provenance, "protocols": protocols, "seeds": seeds}),
              os.path.join(out, "manifest.json"))
    report = {"provenance": provenance, "trial_alignment": data["alignment"]}

    # E. label distribution
    sentence_labels = samples.drop_duplicates("sentence_id")["label_id"].to_numpy()
    report["label_distribution"] = {
        "trials": sanity.label_distribution_baselines(samples["label_id"].to_numpy()),
        "sentences": sanity.label_distribution_baselines(sentence_labels),
    }
    print("label baselines:", json.dumps(report["label_distribution"]["trials"]))

    # F. collapse diagnostics
    report["collapse"] = {key: sanity.collapse_diagnostics(value) for key, value in data["neurolm"].items()}
    if "handcrafted" in data:
        hand = data["handcrafted"]
        filled = np.where(np.isfinite(hand), hand, np.nanmedian(hand, axis=0))
        report["collapse"]["handcrafted"] = sanity.collapse_diagnostics(np.nan_to_num(filled))
    for key, value in report["collapse"].items():
        if value["warnings"]:
            print(f"collapse check {key}: {value['warnings']}")

    report["protocols"] = {}
    for protocol in protocols:
        params = probe["protocols"][protocol]
        block = {}
        observed = run_probe(X, samples, protocol=protocol, classifier=clf, seeds=seeds, protocol_params=params)
        block["observed"] = observed["summary"]
        obs = observed["summary"]["macro_f1"]["mean"]
        print(f"[{protocol}] observed {probe['primary_feature']} {clf}: {obs:.4f}")

        # A. label permutation nulls
        block["label_permutation_sentence"] = sanity.permutation_null(
            X, samples, protocol=protocol, classifier=clf, seeds=seeds, n_permutations=n_perm,
            observed=obs, level="sentence", protocol_params=params, seed=11)
        if protocol == probe["primary_protocol"]:
            block["label_permutation_trial"] = sanity.permutation_null(
                X, samples, protocol=protocol, classifier=clf, seeds=seeds, n_permutations=n_trial_perm,
                observed=obs, level="trial", protocol_params=params, seed=13)
        print(f"[{protocol}] sentence-level null {block['label_permutation_sentence']['null_mean']:.4f}, "
              f"p = {block['label_permutation_sentence']['p_value']:.4f}")

        # B. shuffled sentence association
        shuffled = run_probe(X, samples, protocol=protocol, classifier=checks["association_classifier"],
                             seeds=seeds, protocol_params=params,
                             feature_transform=sanity.shuffle_sentence_association)
        save_probe_result(shuffled, os.path.join(out, protocol, "shuffled_association"), provenance,
                          probe["bootstrap"])
        block["shuffled_association"] = shuffled["summary"]

        # D. Gaussian random features
        random_X = sanity.gaussian_features(len(samples), X.shape[1], checks["random_feature_seed"])
        rand = run_probe(random_X, samples, protocol=protocol, classifier=clf, seeds=seeds, protocol_params=params)
        save_probe_result(rand, os.path.join(out, protocol, "random_gaussian"), provenance, probe["bootstrap"])
        block["random_gaussian"] = rand["summary"]

        # duration-only confound baseline
        duration = run_probe(sanity.duration_features(samples), samples, protocol=protocol, classifier=clf,
                             seeds=seeds, protocol_params=params)
        save_probe_result(duration, os.path.join(out, protocol, "duration_only"), provenance, probe["bootstrap"])
        block["duration_only"] = duration["summary"]
        report["protocols"][protocol] = block
        save_json(report, os.path.join(out, "sanity_summary.json"))

    # C. subject identity
    subjects = sorted(samples["subject_id"].unique())
    samples["subject_code"] = samples["subject_id"].map({s: i for i, s in enumerate(subjects)})
    subject_features = {probe["primary_feature"]: X, "gpt__mean": feature_matrix(data["neurolm"], "gpt__mean")}
    if "handcrafted" in data:
        subject_features["handcrafted"] = data["handcrafted"]
    report["subject_id"] = {"chance_accuracy": 1.0 / len(subjects), "n_subjects": len(subjects),
                            "protocol": checks["subject_id_protocol"]}
    for name, matrix in subject_features.items():
        result = run_probe(matrix, samples, protocol=checks["subject_id_protocol"], classifier="logreg",
                           seeds=seeds[:3], protocol_params=probe["protocols"][checks["subject_id_protocol"]],
                           target="subject_code", class_names=subjects, keep_predictions=False)
        report["subject_id"][name] = result["summary"]
        print(f"subject-ID from {name}: accuracy {result['summary']['accuracy']['mean']:.3f} "
              f"(chance {1 / len(subjects):.3f})")
    report["runtime_s"] = time.time() - started
    save_json(report, os.path.join(out, "sanity_summary.json"))
    pd.DataFrame([
        {"protocol": p, "check": k, "macro_f1_mean": v["macro_f1"]["mean"] if "macro_f1" in v else v.get("null_mean"),
         "p_value": v.get("p_value")}
        for p, block in report["protocols"].items() for k, v in block.items() if isinstance(v, dict)
    ]).to_csv(os.path.join(out, "sanity_table.csv"), index=False)
    print(f"sanity checks done in {time.time() - started:.0f}s -> {out}")


if __name__ == "__main__":
    main()
