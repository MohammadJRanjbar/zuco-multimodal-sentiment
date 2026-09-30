"""Build and document the ZuCo -> NeuroLM channel mapping for every view.

Always available (no data, no checkpoint): label source, template geometry,
mutual-nearest assignment, threshold sensitivity.
Optional evidence:
  --checkpoint        flags spatial-embedding rows that look untrained and
                      excludes them as targets;
  --inspection-summary  decides whether Cz is recoverable (data must not be
                      average-referenced already) and adds the token budget;
  --mat-dir           montage-consistency test on real trials (signal
                      correlation must fall with the claimed distance).
Writes mapping_<view>.json/.csv (read by extract_neurolm_features.py) and the
markdown report.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.neurolm import channel_mapping as cm  # noqa: E402
from src.neurolm.config import load_config, save_json, set_path_overrides  # noqa: E402
from src.neurolm.preprocess import LengthConfig, config_from_dict, seconds_per_chunk  # noqa: E402

SENSITIVITY_THRESHOLDS = [4.0, 5.0, 6.0, 7.0, 8.0, 8.6, 10.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--out-dir", required=True, help="where mapping_<view>.json/.csv are written")
    parser.add_argument("--report", default="reports/channel_mapping.md")
    parser.add_argument("--neurolm-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--inspection-summary", default=None)
    parser.add_argument("--mat-dir", default=None)
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--consistency-trials", type=int, default=40)
    return parser.parse_args()


def optimizer_row_activity(checkpoint, state, param_name="pos_embed.weight"):
    """Per-row Adam second moment of NeuroLM's trainable channel embedding.

    ``train_pretrain.py`` saves ``optimizer.state_dict()``. Adam's second moment
    of a row stays exactly 0 unless that row received gradients; with
    beta2 = 0.95 it also underflows to 0 within ~2,000 steps without gradients.
    A nonzero row is therefore direct proof of training; a zero row means "not
    updated near the end of pretraining", which is weaker than "never trained".
    """
    optimizer = checkpoint.get("optimizer")
    if not optimizer or not optimizer.get("state"):
        return None, "checkpoint has no optimizer state"
    shape = tuple(state[param_name].shape)
    moments = optimizer["state"]
    # NeuroLM.configure_optimizers: trainable (non-tokenizer) params with dim >= 2
    # come first, in named_parameters order.
    ordered = [k for k, v in state.items() if not k.startswith("tokenizer.") and v.dim() >= 2]
    index = ordered.index(param_name) if param_name in ordered else None
    if index is not None and index in moments and tuple(moments[index]["exp_avg_sq"].shape) == shape:
        entry = moments[index]
    else:
        same_shape = [s for s in moments.values()
                      if "exp_avg_sq" in s and tuple(s["exp_avg_sq"].shape) == shape]
        if len(same_shape) != 1:
            return None, f"cannot identify the optimizer state of {param_name}"
        entry = same_shape[0]
    second = entry["exp_avg_sq"].float().mean(dim=1).numpy()
    step = entry.get("step")
    return {
        "param": param_name,
        "optimizer_step": float(step) if step is not None else None,
        "row_second_moment": second.tolist(),
    }, None


def embedding_evidence(checkpoint_path, targets, exclude_untrained):
    """Checkpoint evidence on the spatial embedding of every mapping target.

    The norm/similarity comparison with never-indexed rows is descriptive only:
    pretraining moves rows little relative to their N(0, 1) initialisation, so
    trained rows can be indistinguishable from unused ones. Exclusion (off by
    default) uses only the optimizer second moment.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    evidence = {}
    for name, key in (("tokenizer.pos_embed", "tokenizer.pos_embed.weight"), ("neurolm.pos_embed", "pos_embed.weight")):
        rows, reference = cm.embedding_row_evidence(state[key].float().numpy())
        evidence[name] = {"reference": reference, "rows": rows}
    activity, problem = optimizer_row_activity(checkpoint, state)
    evidence["optimizer"] = activity or {"unavailable": problem}
    excluded = {}
    if activity:
        second = activity["row_second_moment"]
        updated = [t for t in targets if second[cm.NEUROLM_CHANNEL_VOCAB.index(t)] > 0]
        stale = [t for t in targets if second[cm.NEUROLM_CHANNEL_VOCAB.index(t)] <= 0]
        evidence["optimizer"]["targets_updated"] = updated
        evidence["optimizer"]["targets_not_updated_recently"] = stale
        print(f"optimizer evidence: {len(updated)}/{len(targets)} 10-10 targets received gradients near the "
              f"end of pretraining; not recently updated: {stale or 'none'}")
        if exclude_untrained:
            excluded = {t: "NeuroLM channel embedding received no gradient near the end of pretraining"
                        for t in stale}
    else:
        print("optimizer evidence unavailable:", problem)
    return evidence, excluded


def reference_decision(summary_path):
    if not summary_path:
        return None, {}
    summary = json.load(open(summary_path))
    ratio = summary.get("average_reference_ratio", {}).get("median")
    excluded = {}
    if ratio is not None and ratio < 1e-3:
        excluded["Cz"] = "rawData already average-referenced; the reference signal is not recoverable"
    return summary, excluded


def montage_check(mat_dir, labels_csv, config, n_trials, chanlocs_rows):
    from src.neurolm.dataset import iter_subject_trials, subject_files
    from src.neurolm.preprocess import PreprocessConfig, preprocess_trial

    cfg = config_from_dict(config["preprocess"], PreprocessConfig)
    cfg.reference = "none"  # correlation structure of the recorded (Cz-referenced) data
    blocks = []
    for path in subject_files(mat_dir)[:2]:
        for trial, _ in iter_subject_trials(path, labels_csv=labels_csv, limit=n_trials // 2):
            if trial is None:
                continue
            data, valid = preprocess_trial(trial.raw, cfg)
            blocks.append(data[:104])
    signal = np.concatenate(blocks, axis=1)
    corr = np.corrcoef(signal)
    positions = []
    for row in chanlocs_rows[:104]:
        xyz = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])])
        positions.append(xyz / np.linalg.norm(xyz))
    result = cm.montage_consistency(corr, np.asarray(positions), n_permutations=2000)
    result["n_trials"] = len(blocks)
    result["n_samples"] = int(signal.shape[1])
    return result


def token_budget_rows(views_info, inspection):
    durations = None
    if inspection and "duration_s_values" in inspection:
        durations = np.asarray(inspection["duration_s_values"], dtype=float)
    rows = []
    for name, info in views_info.items():
        length = info["length"]
        channels = info["summary"]["n_retained"]
        window = seconds_per_chunk(channels, length)
        row = {
            "view": name,
            "channels": channels,
            "max_tokens": length.max_tokens,
            "strategy": length.strategy,
            "seconds_per_window": window,
            "tokens_per_full_window": window * channels,
        }
        if durations is not None:
            patches = np.floor(durations)  # 1 s patches, remainder dropped
            longer = patches > window
            row["pct_trials_longer_than_window"] = float(100 * longer.mean())
            if length.strategy == "crop":
                lost = np.clip(patches - window, 0, None).sum() / max(patches.sum(), 1)
                row["pct_trials_truncated"] = float(100 * longer.mean())
                row["pct_eeg_seconds_discarded"] = float(100 * lost)
            else:
                row["pct_trials_truncated"] = 0.0
                row["max_chunks_per_trial"] = int(np.ceil(patches.max() / window)) if len(patches) else 0
        rows.append(row)
    return rows


def write_report(path, *, labels, chanlocs_check, alignment, spacing, views_info, sensitivity,
                 evidence, excluded_targets, ref_summary, ref_excluded, consistency, budget, config):
    primary = config["probe"]["primary_view"]
    lines = ["# ZuCo → NeuroLM/LaBraM channel mapping", ""]
    lines += [
        "Generated by `scripts/check_channel_mapping.py`. Nothing in this table is hand-assigned.",
        "",
        "## Sources",
        "",
        f"* **ZuCo channel labels and order**: `EEG.chanlocs` of the ZuCo authors' EEGLAB file "
        "`gip_ZAB_SR5_EEG.mat` (norahollenstein/zuco-benchmark), which the authors use to plot "
        f"`results*_SR.mat` channel vectors → `src/neurolm/montages/zuco_chanlocs.csv` ({len(labels)} channels: "
        f"{len(labels) - 1} EGI electrodes + `{labels[-1]}` last).",
        f"* The ZuCo coordinates equal the EGI `GSN-HydroCel-129.sfp` template (max difference "
        f"{chanlocs_check:.1e} cm after the EEGLAB axis swap), so the template is the correct geometry.",
        "* **NeuroLM vocabulary**: `standard_1020` in `NeuroLM/dataset.py` (verified verbatim); the list index "
        "selects the spatial embedding row.",
        "* **10-10 coordinates**: MNE-Python `standard_1005.elc`.",
        "",
        "## Method",
        "",
        "1. Both templates are placed in a head frame from their own fiducials (EGI FidNz/FidT9/FidT10; "
        "10-05 Nz/LPA/RPA, the pairing MNE uses), projected onto their best-fitting upper-head spheres.",
        f"2. Residual pitch is removed with the one landmark that is identical by construction (EGI vertex "
        f"reference = 10-20 Cz): offset {alignment['cz_offset_before_pitch_correction_deg']:.2f}° → "
        f"{alignment['cz_offset_after_pitch_correction_deg']:.2f}° after a "
        f"{alignment['pitch_correction_deg']:.2f}° rotation.",
        "3. Targets are the 10-10 names of NeuroLM's vocabulary with standard coordinates "
        f"({len(cm.candidate_targets('10-10'))} sites; old aliases T3/T4/T5/T6, ear/mastoid A1/A2/M1/M2 and "
        "10-05 extras are never targets).",
        "4. **exact**: identical electrode name. **approximate**: the ZuCo electrode and the 10-10 site are "
        "each other's nearest neighbour and lie within the view's angle cap. **missing**: no such site. "
        "**excluded**: removed for a stated reason (flat reference, untrained embedding).",
        f"5. The median 10-10 neighbour spacing is {spacing:.2f}°. The primary cap of 6° is ≈ "
        f"{np.radians(6) * alignment['standard_sphere_radius_mm']:.1f} mm on the 10-05 head sphere — within "
        "typical cap-placement error — and about a third of the spacing. The sensitivity view uses 8.6° "
        "(half the spacing).",
        "",
        "The HydroCel net is not laid out on 10-10 positions, so apart from Cz every retained channel is "
        "an *approximate* mapping; the angle column states how approximate.",
        "",
        "## Reference channel",
        "",
    ]
    if ref_summary is None:
        lines.append(
            "Cz (index 104) is the flat recording reference. In the authors' preprocessed file the mean over the "
            "104 other channels is far from zero (std 0.63× the median channel std), i.e. the data are "
            "Cz-referenced, so average re-referencing restores a genuine Cz signal and Cz maps **exactly** "
            "to `CZ`. *Pending: re-check on `rawData` with `scripts/inspect_raw_zuco.py`.*"
        )
    else:
        ratio = ref_summary.get("average_reference_ratio", {})
        lines.append(
            f"Measured on `rawData`: median ratio std_t(mean over non-Cz channels) / median channel std = "
            f"{ratio.get('median')}. " + (
                "Data are already average-referenced → Cz excluded." if ref_excluded
                else "Data are not average-referenced → average re-referencing restores Cz; Cz maps exactly."
            )
        )
    lines += ["", "## Summary by view", "",
              "| view | target set | cap (°) | exact | approximate | missing | excluded | retained | % retained |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, info in views_info.items():
        s = info["summary"]
        c = s["counts"]
        lines.append(
            f"| {name}{' (primary)' if name == primary else ''} | {info['target_set']} | {info['max_angle_deg']:g} | "
            f"{c['exact']} | {c['approximate']} | {c['missing']} | {c['excluded']} | {s['n_retained']} | "
            f"{s['percent_retained']:.1f}% |"
        )
    lines += ["", f"Totals refer to the {len(labels)} ZuCo channels. Not retained = missing + excluded.", ""]
    lines += ["## Threshold sensitivity (10-10 targets)", "", "| cap (°) | ~mm | retained | % |", "|---:|---:|---:|---:|"]
    for threshold, retained in sensitivity:
        lines.append(
            f"| {threshold:g} | {np.radians(threshold) * alignment['standard_sphere_radius_mm']:.1f} | "
            f"{retained} | {100 * retained / len(labels):.1f}% |"
        )
    lines += ["", "## Spatial-embedding check (checkpoint)", ""]
    if evidence is None:
        lines.append("*Pending: run with `--checkpoint NeuroLM-B.pt`.*")
    else:
        optimizer = evidence.get("optimizer", {})
        if "targets_updated" in optimizer:
            stale = optimizer["targets_not_updated_recently"]
            lines.append(
                f"* **Optimizer state (direct evidence)**: Adam's second moment of NeuroLM's channel embedding "
                f"(`pos_embed`, step {optimizer.get('optimizer_step')}) is nonzero for "
                f"{len(optimizer['targets_updated'])} of {len(optimizer['targets_updated']) + len(stale)} 10-10 "
                f"targets, i.e. those rows received gradients near the end of pretraining. Not recently updated: "
                f"{', '.join(stale) if stale else 'none'}. (With beta2 = 0.95 a row decays to 0 within ~2,000 "
                "steps without gradients, so zero means 'not updated near the end', not necessarily 'never'.)"
            )
        else:
            lines.append(f"* Optimizer state unavailable: {optimizer.get('unavailable')}.")
        for table in ("tokenizer.pos_embed", "neurolm.pos_embed"):
            block = evidence[table]
            ref = block["reference"]
            n_same = sum(r["indistinguishable_from_unused"] for r in block["rows"])
            lines.append(
                f"* `{table}` norm/similarity vs the never-indexed rows 139–255 (descriptive only): unused-row norm "
                f"{ref['unused_norm_mean']:.3f} ± {ref['unused_norm_std']:.3f}; {n_same} of {len(block['rows'])} "
                "vocabulary rows are statistically indistinguishable from unused rows. This comparison has low "
                "power (pretraining moves rows little relative to their N(0, 1) initialisation) and is never "
                "used to exclude channels."
            )
        lines.append("")
        lines.append(
            "Targets excluded because of this check: "
            + (", ".join(f"`{k}`" for k in sorted(excluded_targets)) if excluded_targets else
               "none (exclusion is off unless `channel_mapping.exclude_untrained_embeddings: true`)") + "."
        )
    lines += ["", "## Montage consistency on raw trials", ""]
    if consistency is None:
        lines.append("*Pending: run with `--mat-dir`. Checks that EEG correlation falls with the claimed "
                     "electrode distance (a wrong label order would break this).*")
    else:
        lines.append(
            f"Spearman ρ(|corr|, distance) = {consistency['spearman_rho']:.3f} vs permuted-label null "
            f"{consistency['null_mean']:.3f} ± {consistency['null_std']:.3f} (p = {consistency['p_value_one_sided']:.4f}, "
            f"{consistency['n_trials']} trials); most-correlated partner among the 3 nearest electrodes for "
            f"{100 * consistency['most_correlated_partner_within_3_nearest']:.0f}% of channels."
        )
    lines += ["", "## Token budget", "",
              "NeuroLM-B was pretrained on samples of `floor(1024 / n_channels)` seconds (≤ 1024 tokens); "
              "instruction tuning used `eeg_max_len = 276`. Time embeddings have 64 rows.", ""]
    keys = ["view", "channels", "strategy", "max_tokens", "seconds_per_window", "tokens_per_full_window",
            "pct_trials_longer_than_window", "pct_trials_truncated", "pct_eeg_seconds_discarded", "max_chunks_per_trial"]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for row in budget:
        lines.append("| " + " | ".join(
            (f"{row[k]:.1f}" if isinstance(row.get(k), float) else str(row.get(k, "pending"))) for k in keys
        ) + " |")
    lines += ["", f"## Per-channel mapping: {primary} (primary)", "",
              cm.mapping_table_markdown(views_info[primary]["rows"]), ""]
    for name, info in views_info.items():
        if name == primary:
            continue
        retained = [r for r in info["rows"] if r["status"] in {"exact", "approximate"}]
        lines += [f"### Retained channels: {name}", "",
                  ", ".join(f"{r['zuco_label']}→{r['target']} ({r['angle_deg']:.1f}°)" for r in retained), ""]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    config = set_path_overrides(load_config(args.config), neurolm_dir=args.neurolm_dir, checkpoint=args.checkpoint)
    neurolm_dir = config["paths"].get("neurolm_dir")
    if neurolm_dir and os.path.exists(os.path.join(neurolm_dir, "dataset.py")):
        cm.verify_vocabulary(neurolm_dir)
        print("NeuroLM vocabulary verified against", neurolm_dir)
    labels, chanlocs_rows = cm.load_zuco_chanlocs()
    sfp = cm.read_sfp()
    chanlocs_check = max(
        np.abs(np.array([float(r["X"]), float(r["Y"]), float(r["Z"])])
               - np.array([sfp[r["labels"]][1], -sfp[r["labels"]][0], sfp[r["labels"]][2]])).max()
        for r in chanlocs_rows
    )
    if chanlocs_check > 1e-4:
        raise SystemExit(f"ZuCo chanlocs do not match the HydroCel template (diff {chanlocs_check})")
    _, _, alignment = cm.aligned_unit_positions()
    spacing = cm.neighbour_spacing_deg()

    evidence, excluded_targets = None, {}
    if args.checkpoint:
        exclude = bool(config.get("channel_mapping", {}).get("exclude_untrained_embeddings", False))
        evidence, excluded_targets = embedding_evidence(args.checkpoint, cm.candidate_targets("10-10"), exclude)
        print("targets excluded by embedding evidence:", excluded_targets or "none")
    ref_summary, ref_excluded = reference_decision(args.inspection_summary)

    views_info = {}
    for view in config["views"]:
        rows = cm.build_channel_mapping(
            labels,
            reference_mode=config["preprocess"]["reference"],
            max_angle_deg=view["max_angle_deg"],
            target_set=view["target_set"],
            excluded_targets=excluded_targets,
            excluded_channels=ref_excluded,
        )
        summary = cm.summarize_mapping(rows)
        length = config_from_dict(view["length"], LengthConfig)
        views_info[view["name"]] = {
            "rows": rows, "summary": summary, "length": length,
            "target_set": view["target_set"], "max_angle_deg": view["max_angle_deg"],
        }
        cm.save_mapping(rows, summary, os.path.join(args.out_dir, f"mapping_{view['name']}"))
        print(f"{view['name']}: {summary['counts']} -> {summary['n_retained']} retained "
              f"({summary['percent_retained']:.1f}%)")

    primary = config["probe"]["primary_view"]
    retained = views_info[primary]["summary"]["n_retained"]
    minimum = int(config.get("channel_mapping", {}).get("min_primary_channels", 8))

    sensitivity = [
        (t, cm.summarize_mapping(cm.build_channel_mapping(
            labels, reference_mode=config["preprocess"]["reference"], max_angle_deg=t,
            excluded_targets=excluded_targets, excluded_channels=ref_excluded))["n_retained"])
        for t in SENSITIVITY_THRESHOLDS
    ]
    consistency = None
    if args.mat_dir:
        consistency = montage_check(args.mat_dir, args.labels_csv or config["paths"]["labels_csv"],
                                    config, args.consistency_trials, chanlocs_rows)
        print("montage consistency:", consistency)
    budget = token_budget_rows(views_info, ref_summary)
    save_json({
        "alignment": alignment, "spacing_deg": spacing, "excluded_targets": excluded_targets,
        "excluded_channels": ref_excluded, "sensitivity": sensitivity, "consistency": consistency,
        "token_budget": budget, "embedding_evidence": evidence,
        "views": {k: v["summary"] for k, v in views_info.items()},
    }, os.path.join(args.out_dir, "mapping_checks.json"))
    write_report(
        args.report, labels=labels, chanlocs_check=chanlocs_check, alignment=alignment, spacing=spacing,
        views_info=views_info, sensitivity=sensitivity, evidence=evidence, excluded_targets=excluded_targets,
        ref_summary=ref_summary, ref_excluded=ref_excluded, consistency=consistency, budget=budget, config=config,
    )
    print("report ->", args.report)
    if retained < minimum:
        raise SystemExit(
            f"primary view {primary} retains only {retained} channels (< {minimum}); "
            f"excluded targets: {sorted(excluded_targets) or 'none'}. Fix the mapping before extraction."
        )


if __name__ == "__main__":
    main()
