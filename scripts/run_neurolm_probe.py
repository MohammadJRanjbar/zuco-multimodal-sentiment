"""Frozen-feature EEG-only sentiment probes under the three split protocols.

For every protocol in the config: handcrafted features (logistic regression,
linear SVM) and frozen NeuroLM features (logistic regression, linear SVM,
shallow MLP, linear layer) on the SAME trials, plus predeclared secondary
feature sets and views. Saves per-seed metrics, predictions, splits,
cluster-bootstrap intervals, and paired NeuroLM-vs-handcrafted tests.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neurolm.config import load_config, run_manifest, save_json, set_path_overrides  # noqa: E402
from src.neurolm.evaluation import paired_comparison, run_probe  # noqa: E402
from src.neurolm.experiment import (  # noqa: E402
    build_probe_data, feature_matrix, load_neurolm_view, save_probe_result,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--cache-dir")
    parser.add_argument("--handcrafted-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--run-tag", default="probe_v1")
    parser.add_argument("--protocols", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="one seed, joint protocol, logistic regression only (pipeline check)")
    parser.add_argument("--skip-secondary", action="store_true")
    return parser.parse_args()


def flat_row(summary, feature, view, n_trials):
    s = summary["summary"]
    row = {
        "protocol": summary["protocol"], "feature": feature, "view": view, "classifier": summary["classifier"],
        "n_trials": n_trials, "n_seeds": len(summary["seeds"]),
    }
    for key in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"):
        row[f"{key}_mean"] = s[key]["mean"]
        row[f"{key}_std"] = s[key]["std"]
    for name, value in s["per_class_f1_mean"].items():
        row[f"f1_{name}"] = value
        row[f"precision_{name}"] = s["per_class_precision_mean"][name]
        row[f"recall_{name}"] = s["per_class_recall_mean"][name]
    row["confusion_matrix_sum_over_seeds"] = s["confusion_matrix_sum"]
    for cluster, block in summary["bootstrap"].items():
        tag = "sent" if cluster == "sentence_id" else "subj"
        row[f"macro_f1_ci95_{tag}"] = block["macro_f1"]["ci95"]
    aggregated = summary.get("sentence_aggregated")
    if aggregated:
        row["sentence_aggregated_macro_f1_mean"] = aggregated["macro_f1"]["mean"]
        row["sentence_aggregated_macro_f1_std"] = aggregated["macro_f1"]["std"]
    row["runtime_s"] = summary["runtime_s"]
    return row


def main():
    args = parse_args()
    config = set_path_overrides(load_config(args.config), cache_dir=args.cache_dir,
                                handcrafted_dir=args.handcrafted_dir, results_dir=args.results_dir)
    probe = config["probe"]
    protocols = args.protocols or probe["protocols_to_run"]
    seeds = args.seeds or probe["seeds"]
    classifiers = probe["classifiers"]
    if args.smoke:
        protocols, seeds = ["joint"], seeds[:1]
        classifiers = {"handcrafted": ["logreg"], "neurolm": ["logreg"]}
    run_dir = os.path.join(config["paths"]["results_dir"], args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    started = time.time()
    data = build_probe_data(config)
    samples = data["samples"]
    print(f"trials: {data['alignment']}")
    provenance = {
        "experiment": config["experiment"],
        "view": probe["primary_view"],
        "view_path": data["view_path"],
        "checkpoint": data["payload"]["checkpoint"],
        "channel_mapping": data["payload"]["channels"],
        "preprocess": data["payload"]["preprocess"],
        "length": data["payload"]["length"],
        "trial_alignment": data["alignment"],
    }
    save_json(run_manifest(config, {**provenance, "protocols": protocols, "seeds": seeds}),
              os.path.join(run_dir, "manifest.json"))
    samples[["sample_id", "subject_id", "sentence_id", "label_id", "duration_s"]].to_csv(
        os.path.join(run_dir, "samples.csv"), index=False)

    jobs = []
    if "handcrafted" in data:
        jobs += [("handcrafted", data["handcrafted"], clf, "classical_cache") for clf in classifiers["handcrafted"]]
    primary = feature_matrix(data["neurolm"], probe["primary_feature"])
    jobs += [(probe["primary_feature"], primary, clf, probe["primary_view"]) for clf in classifiers["neurolm"]]
    if not args.smoke and not args.skip_secondary:
        for key in probe["secondary_features"]:
            jobs.append((key, feature_matrix(data["neurolm"], key), probe["primary_classifier"], probe["primary_view"]))

    rows, paired_rows = [], []
    for protocol in protocols:
        params = probe["protocols"][protocol]
        clusters = ("sentence_id",) + (("subject_id",) if protocol in {"subject", "joint"} else ())
        predictions = {}
        for feature, X, clf, view in jobs:
            t0 = time.time()
            result = run_probe(X, samples, protocol=protocol, classifier=clf, seeds=seeds, protocol_params=params)
            out_dir = os.path.join(run_dir, protocol, f"{view}__{feature}__{clf}")
            summary, preds = save_probe_result(
                result, out_dir, {**provenance, "feature": feature, "feature_dim": int(X.shape[1])},
                probe["bootstrap"], clusters)
            predictions[(feature, clf)] = preds
            rows.append(flat_row(summary, feature, view, len(samples)))
            print(f"[{protocol}] {feature:>24s} {clf:>6s}: macro-F1 {summary['summary']['macro_f1']['mean']:.4f}"
                  f" ± {summary['summary']['macro_f1']['std']:.4f} ({time.time() - t0:.0f}s)")
        for clf in classifiers["neurolm"]:
            base_clf = clf if clf in classifiers.get("handcrafted", []) else "logreg"
            if ("handcrafted", base_clf) not in predictions:
                continue
            for cluster in clusters:
                comparison = paired_comparison(
                    predictions[("handcrafted", base_clf)], predictions[(probe["primary_feature"], clf)],
                    cluster=cluster, n_boot=probe["bootstrap"]["n_boot"], n_perm=probe["bootstrap"]["n_perm"])
                paired_rows.append({"protocol": protocol, "baseline": f"handcrafted__{base_clf}",
                                    "candidate": f"{probe['primary_feature']}__{clf}", **comparison})

    if not args.smoke and not args.skip_secondary:
        protocol = probe["primary_protocol"]
        for view_name in probe["secondary_views"]:
            _, metadata, features, _ = load_neurolm_view(config, view_name)
            index = pd.Series(np.arange(len(metadata)), index=metadata["sample_id"])
            common = samples["sample_id"][samples["sample_id"].isin(index.index)]
            view_samples = samples[samples["sample_id"].isin(common)].reset_index(drop=True)
            X = feature_matrix(features, probe["primary_feature"])[index.loc[view_samples["sample_id"]].to_numpy()]
            result = run_probe(X, view_samples, protocol=protocol, classifier=probe["primary_classifier"],
                               seeds=seeds, protocol_params=probe["protocols"][protocol])
            out_dir = os.path.join(run_dir, protocol, f"{view_name}__{probe['primary_feature']}__{probe['primary_classifier']}")
            summary, _ = save_probe_result(result, out_dir, {**provenance, "view": view_name,
                                                             "feature": probe["primary_feature"]},
                                           probe["bootstrap"], ("sentence_id", "subject_id"))
            rows.append(flat_row(summary, probe["primary_feature"], view_name, len(view_samples)))
            print(f"[{protocol}] view {view_name}: macro-F1 {summary['summary']['macro_f1']['mean']:.4f}")

    tables = os.path.join(run_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(tables, "probe_summary.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(tables, "paired_neurolm_vs_handcrafted.csv"), index=False)
    save_json({"rows": rows, "paired": paired_rows, "runtime_s": time.time() - started},
              os.path.join(tables, "probe_summary.json"))
    print(f"done in {time.time() - started:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
