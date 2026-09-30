"""Shared loading and saving for the probe and sanity runners."""

import os

import numpy as np
import pandas as pd

from .cache import find_view, load_view
from .config import save_json
from .dataset import load_handcrafted_trials
from .evaluation import cluster_bootstrap, predictions_frame, sentence_aggregate
from .preprocess import LengthConfig, PreprocessConfig, config_from_dict


def expected_payload(config, view):
    """Payload fields that can be checked without the checkpoint file."""
    preprocess = config_from_dict(config["preprocess"], PreprocessConfig).to_dict()
    length = config_from_dict(view["length"], LengthConfig).to_dict()
    return {
        "preprocess": preprocess,
        "length": length,
        "poolings": list(config["encoder"]["poolings"]),
        "representations": list(config["encoder"]["representations"])
        + [f"gpt_L{layer}" for layer in config["encoder"].get("gpt_layers", [])],
        "precision": config["encoder"]["precision"],
    }


def view_by_name(config, name):
    for view in config["views"]:
        if view["name"] == name:
            return view
    raise KeyError(f"view {name!r} is not defined in the config")


def feature_matrix(features, key):
    """``a+b`` concatenates pooled features, e.g. ``tokenizer__mean+max``."""
    if key in features:
        return features[key]
    if "+" in key:
        head, *extra = key.split("+")
        representation = head.split("__")[0]
        parts = [features[head]] + [features[f"{representation}__{pooling}"] for pooling in extra]
        return np.concatenate(parts, axis=1)
    raise KeyError(f"feature {key!r} not in cache (available: {sorted(features)})")


def load_neurolm_view(config, name, cache_dir=None):
    view = view_by_name(config, name)
    cache_dir = cache_dir or config["paths"]["cache_dir"]
    path = find_view(cache_dir, name, expected=expected_payload(config, view))
    metadata, features, stored = load_view(path)
    return path, metadata, features, stored


def build_probe_data(config, cache_dir=None, handcrafted_dir=None):
    """Align NeuroLM (primary view) and handcrafted features on common trials."""
    primary = config["probe"]["primary_view"]
    view_path, metadata, features, stored = load_neurolm_view(config, primary, cache_dir)
    usable = metadata["amplitude_ok"].astype(bool).to_numpy() if "amplitude_ok" in metadata else np.ones(len(metadata), bool)
    handcrafted_dir = handcrafted_dir or config["paths"].get("handcrafted_dir")
    hand_table, hand_X = (None, None)
    if handcrafted_dir:
        hand_table, hand_X = load_handcrafted_trials(handcrafted_dir)
    ids = metadata.loc[usable, "sample_id"]
    if hand_table is not None:
        ids = ids[ids.isin(set(hand_table["sample_id"]))]
    common = sorted(ids)
    neuro_index = pd.Series(np.arange(len(metadata)), index=metadata["sample_id"])
    rows = neuro_index.loc[common].to_numpy()
    samples = metadata.iloc[rows].reset_index(drop=True)
    data = {
        "samples": samples,
        "neurolm": {key: value[rows] for key, value in features.items()},
        "view_path": view_path,
        "payload": stored["payload"],
        "alignment": {
            "neurolm_trials": int(len(metadata)),
            "neurolm_amplitude_excluded": int((~usable).sum()),
            "handcrafted_trials": int(len(hand_table)) if hand_table is not None else None,
            "common_trials": int(len(common)),
        },
    }
    if hand_table is not None:
        hand_index = pd.Series(np.arange(len(hand_table)), index=hand_table["sample_id"])
        hand_rows = hand_index.loc[common].to_numpy()
        labels_agree = (hand_table["label_id"].to_numpy()[hand_rows] == samples["label_id"].to_numpy()).all()
        if not labels_agree:
            raise ValueError("handcrafted and NeuroLM labels disagree for some trials")
        data["handcrafted"] = hand_X[hand_rows]
    return data


def save_probe_result(result, out_dir, provenance, bootstrap_cfg, clusters=("sentence_id",)):
    """Write per-seed JSON + predictions CSV, and a summary with CIs."""
    os.makedirs(out_dir, exist_ok=True)
    predictions = predictions_frame(result)
    for record in result["per_seed"]:
        seed = record["seed"]
        record["predictions"].drop(columns=["true_id", "predicted_id", "seed"]).to_csv(
            os.path.join(out_dir, f"predictions_seed_{seed}.csv"), index=False
        )
        save_json({
            **provenance,
            "protocol": result["protocol"],
            "classifier": result["classifier"],
            "seed": seed,
            "metrics": record["metrics"],
            "confusion_matrix": record["metrics"]["confusion_matrix"],
            "splits": [
                {k: fold[k] for k in ("fold", "n_train", "n_val", "n_test", "subjects", "sentences")}
                for fold in record["folds"]
            ],
            "fold_selection": [
                {"fold": fold["fold"], "selection": fold["selection"],
                 "test_macro_f1": fold["test_macro_f1"], "test_accuracy": fold["test_accuracy"]}
                for fold in record["folds"]
            ],
            "runtime_s": record["runtime_s"],
        }, os.path.join(out_dir, f"seed_{seed}.json"))
    intervals = {
        cluster: cluster_bootstrap(predictions, cluster=cluster, n_boot=bootstrap_cfg["n_boot"])
        for cluster in clusters
    }
    summary = {
        **provenance,
        "protocol": result["protocol"],
        "classifier": result["classifier"],
        "seeds": result["seeds"],
        "summary": result["summary"],
        "bootstrap": intervals,
        "sentence_aggregated": sentence_aggregate(predictions) if "prob_negative" in predictions else None,
        "runtime_s": float(sum(r["runtime_s"] for r in result["per_seed"])),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    predictions.to_csv(os.path.join(out_dir, "predictions_all_seeds.csv"), index=False)
    return summary, predictions
