"""Extract frozen NeuroLM embeddings for every subject x sentence trial.

raw ZuCo EEG -> trial-local preprocessing -> channel mapping -> NeuroLM tokens
-> frozen encoder (VQ tokenizer + NeuroLM-B GPT) -> pooled vector per trial.

All views of the config are produced from one pass over the .mat files.
Subjects already cached for every view are skipped, so an interrupted Colab
run resumes. ``--smoke`` runs one subject and ten sentences into a separate
cache root and prints the shapes.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

import numpy as np  # noqa: E402

from src.labels import label_lookup  # noqa: E402
from src.neurolm import cache  # noqa: E402
from src.neurolm.config import load_config, run_manifest, save_json, set_path_overrides  # noqa: E402
from src.neurolm.dataset import LANGUAGE, iter_subject_trials, subject_files, subject_from_path  # noqa: E402
from src.neurolm.encoder import NeuroLMEncoder, pool_trials  # noqa: E402
from src.neurolm.preprocess import (  # noqa: E402
    LengthConfig, PreprocessConfig, amplitude_ok, config_from_dict, preprocess_trial, tokenize,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--mat-dir")
    parser.add_argument("--labels-csv")
    parser.add_argument("--neurolm-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-dir")
    parser.add_argument("--mapping-dir", required=True, help="output folder of check_channel_mapping.py")
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trial-batch", type=int, default=16, help="trials encoded together")
    parser.add_argument("--smoke", action="store_true", help="1 subject, 10 sentences, separate cache root")
    return parser.parse_args()


def load_view_plan(mapping_dir, view):
    stored = json.load(open(os.path.join(mapping_dir, f"mapping_{view['name']}.json")))
    rows = [r for r in stored["rows"] if r["status"] in {"exact", "approximate"}]
    if not rows:
        reasons = {}
        for row in stored["rows"]:
            reasons[row["reason"].split(":")[0]] = reasons.get(row["reason"].split(":")[0], 0) + 1
        raise SystemExit(
            f"view {view['name']} retains no channels in {mapping_dir}. Status reasons: {reasons}. "
            "Re-run scripts/check_channel_mapping.py (notebook step 6) and read reports/channel_mapping.md."
        )
    plan = [(int(r["zuco_index"]), int(r["vocab_index"])) for r in rows]
    channels = [(r["zuco_label"], r["target"]) for r in rows]
    return plan, channels


def main():
    args = parse_args()
    config = set_path_overrides(
        load_config(args.config), mat_dir=args.mat_dir, labels_csv=args.labels_csv,
        neurolm_dir=args.neurolm_dir, checkpoint=args.checkpoint, cache_dir=args.cache_dir,
    )
    paths = config["paths"]
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = args.subjects
    max_sentences = args.max_sentences
    cache_root = paths["cache_dir"]
    if args.smoke:
        files = subject_files(paths["mat_dir"], subjects)
        subjects = subjects or [subject_from_path(files[0])]
        max_sentences = max_sentences or 10
        cache_root = os.path.join(cache_root, "smoke")

    encoder_cfg = config["encoder"]
    encoder = NeuroLMEncoder(
        paths["neurolm_dir"], paths["checkpoint"], device=device,
        precision=encoder_cfg["precision"], gpt_layers=encoder_cfg.get("gpt_layers", []),
    )
    checkpoint = {**encoder.checkpoint_info, "repo": config["checkpoint"]["repo"],
                  "revision": config["checkpoint"]["revision"]}
    print("checkpoint:", json.dumps({k: checkpoint[k] for k in ("checkpoint_name", "checkpoint_sha256", "n_parameters")}))
    preprocess_cfg = config_from_dict(config["preprocess"], PreprocessConfig)
    poolings = encoder_cfg["poolings"]

    views = []
    for view in config["views"]:
        if args.views and view["name"] not in args.views:
            continue
        plan, channels = load_view_plan(args.mapping_dir, view)
        length = config_from_dict(view["length"], LengthConfig)
        payload = cache.fingerprint_payload(
            checkpoint=checkpoint, channels=channels, preprocess=preprocess_cfg.to_dict(),
            length=length.to_dict(), representations=encoder.representations, poolings=poolings,
            precision=encoder_cfg["precision"],
        )
        path = cache.prepare_view(cache_root, view["name"], payload)
        views.append({"name": view["name"], "plan": plan, "length": length, "path": path, "channels": channels})
        print(f"view {view['name']}: {len(plan)} channels -> {path}")

    lookup = label_lookup(paths["labels_csv"])
    started = time.time()
    log = {"subjects": {}, "views": {v["name"]: v["path"] for v in views}}
    for mat_path in subject_files(paths["mat_dir"], subjects):
        subject = subject_from_path(mat_path)
        if all(cache.part_exists(v["path"], subject) for v in views):
            print(f"skip {subject}: cached for every view")
            continue
        subject_start = time.time()
        pending = {v["name"]: {"meta": [], "features": {}} for v in views}
        batch = []

        def flush():
            for view in views:
                tokens = [item[1][view["name"]] for item in batch]
                pooled = pool_trials(encoder, tokens, poolings, batch_size=encoder_cfg["batch_size"])
                store = pending[view["name"]]
                for (trial, _, record), view_tokens, features in zip(batch, tokens, pooled):
                    if features is None:
                        continue
                    for key, vector in features.items():
                        store["features"].setdefault(key, []).append(vector)
                    store["meta"].append({
                        "sample_id": trial.sample_id,
                        "subject_id": trial.subject_id,
                        "sentence_id": trial.sentence_id,
                        "label": trial.label,
                        "label_id": trial.label_id,
                        "duration_s": record["duration_s"],
                        "num_channels": view_tokens.n_channels,
                        "language": LANGUAGE,
                        "n_patches": view_tokens.n_patches,
                        "n_patches_used": view_tokens.n_patches_used,
                        "n_chunks": len(view_tokens.chunks),
                        "n_tokens": int(sum(len(c[0]) for c in view_tokens.chunks)),
                        "seconds_per_chunk": view_tokens.seconds_per_chunk,
                        "truncated": view_tokens.truncated,
                        "dropped_channels": ",".join(map(str, view_tokens.dropped_channels)),
                        "median_channel_std_uv": view_tokens.median_channel_std_uv,
                        "max_abs_uv": view_tokens.max_abs_uv,
                        "amplitude_ok": amplitude_ok(view_tokens, preprocess_cfg),
                        "nan_fraction": record["nan_fraction"],
                    })
            batch.clear()

        skipped = {}
        for trial, record in iter_subject_trials(mat_path, lookup=lookup, sfreq=preprocess_cfg.source_sfreq,
                                                 limit=max_sentences):
            if trial is None:
                skipped[record["status"]] = skipped.get(record["status"], 0) + 1
                continue
            data, valid = preprocess_trial(trial.raw, preprocess_cfg)
            per_view = {v["name"]: tokenize(data, valid, v["plan"], preprocess_cfg, v["length"]) for v in views}
            if any(not t.chunks for t in per_view.values()):
                skipped["no_full_patch_or_channels"] = skipped.get("no_full_patch_or_channels", 0) + 1
                continue
            batch.append((trial, per_view, record))
            if len(batch) >= args.trial_batch:
                flush()
        if batch:
            flush()
        for view in views:
            store = pending[view["name"]]
            if store["meta"]:
                cache.save_part(view["path"], subject, store["meta"], store["features"])
        n = len(pending[views[0]["name"]]["meta"]) if views else 0
        log["subjects"][subject] = {"trials": n, "skipped": skipped, "seconds": time.time() - subject_start}
        print(f"{subject}: {n} trials embedded, skipped {skipped}, {time.time() - subject_start:.0f}s")

    for view in views:
        metadata, merged = cache.merge_parts(view["path"])
        shapes = {k: list(v.shape) for k, v in merged.items() if k != "sample_id"}
        save_json({
            **run_manifest(config, {"checkpoint": checkpoint, "view": view["name"], "channels": view["channels"]}),
            "n_trials": int(len(metadata)),
            "feature_shapes": shapes,
            "truncated_trials": int(metadata["truncated"].sum()),
            "multi_chunk_trials": int((metadata["n_chunks"] > 1).sum()),
            "amplitude_flagged": int((~metadata["amplitude_ok"].astype(bool)).sum()),
            "runtime_s": time.time() - started,
            "extraction_log": log,
        }, os.path.join(view["path"], "extraction_manifest.json"))
        print(f"{view['name']}: {len(metadata)} trials, features {shapes}")


if __name__ == "__main__":
    main()
